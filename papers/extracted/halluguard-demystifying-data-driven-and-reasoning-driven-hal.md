---
source_pdf: papers/HalluGuard Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs.pdf
slug: halluguard-demystifying-data-driven-and-reasoning-driven-hal
pages: 33
extracted_on: 2026-07-13
---

# HalluGuard Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs

## Page 1

Published as a conference paper at ICLR 2026
HALLUGUARD: DEMYSTIFYING DATA-DRIVEN AND
REASONING-DRIVEN HALLUCINATIONS IN LLMS
Xinyue Zeng∗
CS Department
Virginia Tech
Junhong Lin∗
EECS Department
MIT
Yujun Yan
CS Department
Dartmouth College
Feng Guo
Statistics Department
Virginia Tech
Liang Shi
Statistics Department
Virginia Tech
Jun Wu
CS Department
Michigan State University
Dawei Zhou
CS Department
Virginia Tech
ABSTRACT
The reliability of Large Language Models (LLMs) in high-stakes domains such
as healthcare, law, and scientific discovery is often compromised by hallucina-
tions. These failures typically stem from two sources: data-driven hallucinations
and reasoning-driven hallucinations. However, existing detection methods usu-
ally address only one source and rely on task-specific heuristics, limiting their
generalization to complex scenarios. To overcome these limitations, we introduce
the Hallucination Risk Bound, a unified theoretical framework that formally de-
composes hallucination risk into data-driven and reasoning-driven components,
linked respectively to training-time mismatches and inference-time instabilities.
This provides a principled foundation for analyzing how hallucinations emerge
and evolve. Building on this foundation, we introduce HALLUGUARD, a NTK-
based score that leverages the induced geometry and captured representations of
the NTK to jointly identify data-driven and reasoning-driven hallucinations. We
evaluate HALLUGUARD on 10 diverse benchmarks, 11 competitive baselines, and
9 popular LLM backbones, consistently achieving state-of-the-art performance in
detecting diverse forms of LLM hallucinations. We open-source our proposed
HALLUGUARD model at HalluGuard.
1
INTRODUCTION
Large language models (LLMs) are increasingly deployed in high-stakes domains such as health-
care, law, and scientific discovery (Bommasani et al., 2021; Thirunavukarasu et al., 2023; Ke et al.,
2025). However, adoption in these settings remains cautious, as such domains are highly regulated
and demand strict compliance, interpretability, and safety guarantees (Dennst¨adt et al., 2025; Kat-
tnig et al., 2024). A major barrier is the risk of hallucinations, generated content appears unfaithful
or nonsensical. Such errors can have severe consequences (Dennst¨adt et al., 2025), as the example in
Figure 1, a generated incorrect medical diagnosis may delay treatment or lead to harmful interven-
tions. Therefore, detecting hallucinations is not merely a technical challenge but a prerequisite for
trustworthy deployment, as undetected errors undermine reliability, accountability, and user safety.
Generally, hallucinations in LLMs arise from two primary sources (Ji et al., 2023; Huang et al.,
2025): data-driven hallucinations, which stem from flawed, biased, or incomplete knowledge en-
coded during pre-training or fine-tuning; and reasoning-driven hallucinations, which originate from
inference-time failures such as logical inconsistencies or breakdowns in multi-step reasoning (Zhang
et al., 2023; Zhong et al., 2024). Detection methods broadly split along these two dimensions. Ap-
proaches for data-driven hallucinations often compare outputs against retrieved documents or refer-
ences (Shuster et al., 2021; Min et al., 2023; Ji et al., 2023), or exploit sampling consistency as in
SelfCheckGPT (Manakul et al., 2023). In contrast, methods for reasoning-driven hallucinations rely
on signals of inference-time instability, including probabilistic measures such as perplexity (Ren
∗Equal contribution.
1
arXiv:2601.18753v2  [cs.LG]  2 Mar 2026

## Page 2

Published as a conference paper at ICLR 2026
Figure 1: An illustration of hallucination emerging and evolving in the context of disease diagnosis.
et al., 2023), length-normalized entropy (Malinin & Gales, 2021), semantic entropy (Kuhn et al.,
2023), energy-based scoring (Liu et al., 2020), and RACE (Wang et al., 2025). Others probe internal
representations, for example, Inside (Chen et al., 2024), which applies eigenvalue-based covariance
metrics and feature clipping, ICR Probe (Zhang et al., 2025), which tracks residual-stream updates,
and Shadows in the Attention (Wei et al., 2025), which analyzes representation drift under contextual
perturbations. While these methods shed light on the mechanisms underlying hallucinations, most
remain tailored to a single hallucination type and fail to capture their evolution. Yet growing evi-
dence indicates that data-driven and reasoning-driven hallucinations often evolve during multi-step
generation (Liu et al., 2025; Sun et al., 2025). As shown in Figure 1, it emerges from an initial dis-
ease misclassification and evolves into a distorted diagnosis, delaying treatments and risking fatality.
This gap brings two central questions: (1) How can we develop a unified theoretical understand-
ing of how hallucinations evolve? (2) How can we detect them effectively and efficiently without
relying on external references or task-specific heuristics?
To address these challenges, we propose a unified theoretical framework–Hallucination Risk Bound,
which decomposes the overall hallucination risk into two components: a data-driven term, cap-
turing semantic deviations rooted in inaccurate, imbalanced, or noisy supervision acquired during
model training; and a reasoning-driven term, reflecting instability introduced by inference-time dy-
namics, such as logical missteps or temporal inconsistency. This decomposition not only elucidates
the mechanism behind hallucinations but also reveals how they emerge and evolve. Specifically,
our analysis shows that hallucinations originate from semantic approximation gaps, captured by
representational limits of the model, and are subsequently amplified by unstable rollout dynamics,
evolving across decoding steps. As such, our framework offers a unified theoretical lens for charac-
terizing the emergence and evolution of these hallucinations.
Building on the theoretical foundation, we propose HALLUGUARD, a Neural Tangent Kernel(NTK)-
based score that leverages the induced geometry and captured representations of the NTK to jointly
identify data-driven and reasoning-driven hallucinations. We evaluate HALLUGUARD comprehen-
sively across 10 diverse benchmarks, 11 competitive baselines, and 9 popular LLM backbones.
HALLUGUARD consistently achieves state-of-the-art hallucination detection performance, demon-
strating its efficacy. We open-source our proposed HALLUGUARD model at HalluGuard.
2
PRELIMINARIES
Hallucination Detection.
There are two primary sources of hallucinations in LLMs (Ji et al.,
2023; Huang et al., 2025): data-driven hallucination, which stems from incomplete or biased knowl-
edge encoded during pre-training or fine-tuning, and reasoning-driven hallucination, which arises
from unstable or inconsistent inference dynamics at decoding time. This distinction has implicitly
guided a broad range of detection strategies, which we examine through these two lenses.
For data-driven causes, a recurring signal is elevated predictive uncertainty. A common formulation
adopts the sequence-level negative log-likelihood:
U(y | x, θ) = −1
T
T
X
t=1
log pθ(yt | y<t, x),
(1)
which quantifies the average uncertainty of generating a sequence y = [y1, . . . , yT ] from input x and
θ denotes model parameters. This directly recovers Perplexity (Ren et al., 2023), where low scores
2

## Page 3

Published as a conference paper at ICLR 2026
imply confident predictions, while high scores indicate implausible generations due to weak priors.
To capture more nuanced uncertainty, later methods extend this formulation to multi-sample settings.
The Length-Normalized Entropy (Malinin & Gales, 2021) penalizes dispersion across stochastic
generations Y = {y1, . . . , yK} where K denotes the number of independent stochastic rollouts
sampled from the model for a given input, offering a finer-grained view of model indecision. This
perspective is further enriched by Semantic Entropy (Kuhn et al., 2023), which projects sampled
responses into semantic space, and by energy-based scoring (Liu et al., 2020), which replaces log-
probability with a learned confidence function. Collectively, these methods reflect a progression
from token-level likelihoods to semantically grounded multi-sample uncertainty estimators.
In contrast, reasoning-driven hallucinations arise from brittle inference trajectories, where identical
contexts may yield inconsistent or incoherent outputs. A commonly used measure of such instability
is the cross-sample consistency score:
C(Y | x, θ) = 1
C
K
X
i=1
K
X
j=i+1
sim(yi, yj),
(2)
where C = K · (K−1)
2
, and sim(·, ·) is a similarity function such as ROUGE-L (Lin, 2004), co-
sine similarity, or BLEU (Chen et al., 2023). Low scores reflect diverging generations and un-
stable reasoning. Several reasoning-driven detection methods can be interpreted through this lens.
Early approaches used surface-level lexical overlap metrics (Lin et al., 2022b), while SelfCheck-
GPT (Manakul et al., 2023) advanced this by evaluating factual entailment across responses, and
FActScore (Min et al., 2023) extended this further by comparing outputs to retrieved reference doc-
uments. More recent efforts probe internal signals directly: Inside (Chen et al., 2024) analyzes
the covariance spectrum of embedding representations, and RACE (Wang et al., 2025) diagnoses
instability in multi-step reasoning.
NTK in LLMs.
NTK provides a principled framework for analyzing the training dynamics in
the overparameterized regime characteristic of modern LLMs (Jacot et al., 2018). Formally, for a
network output f(x, θ) with input x and parameters θ, the NTK is defined as:
Θ(x, x′, θ) = ∇θf(x, θ) · ∇θf(x′, θ).
(3)
This kernel Θ(x, x′, θ) quantifies the similarity of training dynamics between inputs x and x′. In the
infinite-width limit, it converges to a deterministic value at initialization and remains nearly constant
throughout training (Lee et al., 2020b). This stability reduces the highly nonlinear optimization of
deep networks to a tractable kernel regression problem. By examining the eigenspectrum of the
NTK, one can probe how internal representations are shaped during training: which features are
prioritized (e.g., syntax versus semantics), how quickly different tasks converge, and why overpa-
rameterized networks generalize effectively to unseen data (Ju et al., 2022). In this way, the NTK
transforms the apparent complexity of LLM optimization into a clear lens on how these models
capture, process, and generalize information (Zeng et al., 2025).
3
METHODOLOGY
3.1
PROBLEM SETTING
Our analysis reveals that hallucination is not a unified failure mode but rather shifts with the task
structure. On the instruction-following Natural benchmark (Wang et al., 2022), 88.9% of the
overall 3499 errors are from logical missteps (reasoning-driven) while 11.1% are factual inaccu-
racies (data-driven). By contrast, on the math-focused MATH-500 (Hendrycks et al., 2021), the
1985 wrong generations are dominated by 1946 reasoning errors (98.1%), with only 19 factual
flaws (1.9%). This contrast highlights that, in practice, hallucinations are rarely pure but often mix-
tures of data-driven bias and reasoning-driven instability-motivating our formal decomposition of
hallucination sources.
Problem Definition.
Let Y denote the discrete space of all possible finite-length textual token sequences. We define
a continuous semantic embedding space Uh ⊆
Rdh equipped with a norm ∥· ∥. Each vector
u ∈Uh represents the semantic representation of a reasoning chain composed of step-wise logical
3

## Page 4

Published as a conference paper at ICLR 2026
statements. We define a task-specific encoder Φ : Y →Uh that maps a discrete textual sequence
into this continuous hypothesis space. In this framework, for an input x with a ground-truth output
sequence y∗∈Y, we define the target semantic representation as u∗:= Φ(y∗) ∈Uh. An LLM
with parameters θ emits a random sequence Y = (Y1, . . . , YT ) ∈Y via the autoregressive decoding
distribution pθ(yt | y<t, x), yielding a predicted semantic representation uh := Φ(Y ) ∈Uh. Thus,
the model’s expected sematic output is defined as E[uh] := EY ∼pθ(·|x)[Φ(Y )].
To analyze inference dynamics, we consider perturbations in a local neighborhood of the decoding
process. Let Rr denote the r-dimensional continuous space of the model’s internal states (e.g., prefix
embeddings or hidden activations). We parameterize a small perturbation by δ ∈Rr, restricted to
a local ℓ2-ball Bρ := {δ ∈Rr : ∥δ∥2 ≤ρ}. Let Pθ(· | x, δ) denote the perturbed decoding
distribution induced by δ. We define the mean semantic response map GY : Rr →Uh, GY (δ) :=
EY ∼Pθ(·|x,δ)[Φ(Y )] with its corresponding inference Jacobian J := DGY (0) ∈Rdh×r. Thus, we
formally define the problem as follows:
Problem 1 (Hallucination Dynamics Characterization).
Given: (1) The target semantic representation u∗:= Φ(y∗) ∈Uh for a ground-truth output y∗∈Y;
(2) the random sequence Y ∈Y emitted via the autoregressive decoding distribution pθ(yt | y<t, x),
yielding a predicted representation uh := Φ(Y ) with expected value E[uh] ; and (3) the inference
constraints defined by a local perturbation δ ∈Rr restricted to the ℓ2-ball Bρ
Find: A formal geometric mechanism to characterize how hallucinations emerge and evolve by
analyzing the Mean Semantic Response Map GY (δ) and the Inference Jacobian J, which captures
the sensitivity of the model’s reasoning trajectory to internal instabilities.
3.2
HALLUCINATION RISK BOUND
To bridge the formal setup with the phenomenon of hallucination, we first disentangle the sources of
hallucinations. Intuitively, hallucinations may arise either from systematic biases in the knowledge
encoded by the model (data-driven) or from instabilities during autoregressive decoding (reasoning-
driven). The following proposition formalizes this idea by decomposing the total hallucination risk
into two components.
We first impose the following assumptions:
A1. (Uh, ∥· ∥) is a finite-dimensional Hilbert space. The encoder Φ : Y →Uh is measurable,
and the random variable Φ(Y ) has finite second moment under the model’s unperturbed
decoding distribution: EY ∼pθ(·|x)

∥Φ(Y )∥2
< ∞. This ensures that the mean semantic
representation E[Φ(Y )] is well-defined in Uh.
A2. Let (Y, dY) be the discrete metric space equipped with edit distance.
The encoder
Φ
:
(Y, dY)
→
(Uh, ∥· ∥) is LΦ-Lipschitz continuous:
∥Φ(y) −Φ(y′)∥
≤
LΦ dY(y, y′)
∀y, y′ ∈Y.
A3. For any perturbation δ in the closed ball Bρ := {δ ∈Rr : ∥δ∥2 ≤ρ}, the mean se-
mantic response map GY (δ) = EY ∼Pθ(·|x,δ)[Φ(Y )] is twice Fr´echet differentiable in a
neighborhood of δ = 0 and admits the expansion GY (δ) = GY (0) + Jδ + R(δ), where
J = DGY (0) ∈Rdh×r and the remainder satisfies ∥R(δ)∥≤1
2H⋆∥δ∥2
2, ∀δ ∈Bρ, for
some constant H⋆> 0.
Proposition 3.1 (Hallucination Risk Decomposition).
Under A1-A3, applying the triangle
inequality yields a natural split of the risk:
∥u∗−uh∥≤∥u∗−E[uh]∥
|
{z
}
data-driven term
+ ∥uh −E[uh]∥
|
{z
}
reasoning-driven term
This decomposition distinguishes errors caused by systematic bias in the learned representation
from those introduced during stochastic rollout.
Characterizing Data-Driven Hallucination.
To quantify the data-driven term, we take inspira-
tion from the NTK, which has proven effective in analyzing training dynamics of overparameterized
models. Here, NTK geometry provides a way to measure how well the model’s representation space
aligns with task generation under small perturbations.
4

## Page 5

Published as a conference paper at ICLR 2026
Let Uh denote the hypothesis subspace accessible to the model under perturbations. By C´ea’s
lemma(C´ea, 1964) with curvature penalty, the data-driven term can be bounded as
∥u∗−E[uh]∥≤Λ
γ
inf
u∈Uh ∥u∗−u∥,
(4)
where γ = λmin(KΦ) is the smallest eigenvalue of the NTK Gram matrix on embedded pertur-
bations KΦ, and Λ ≤∥T ∥, where T : Uh →Uh denotes the operator mapping. Intuitively, the
ratio Λ
γ measures the conditioning of the feature map: well-conditioned NTK spectra allow a closer
approximation to the true generation.
Thus, the ratio can be further controlled in terms of pretraining-finetuning mismatch:
Λ
γ
≤1 + kpt logO(P, L) + k · ϵmismatch
Signalk
,
(5)
where logO(P, L) is a complexity term from parameter count P and prompt length L, ϵmismatch
denotes the Wasserstein distance between prompt and query distributions, Signalk measures task-
aligned energy in the top-k eigenspace. kpt and k are task and model-dependent constants. Thus,
data-driven hallucinations grow when the mismatch is large or when the task signal is weak.
Characterizing
Reasoning-Driven
Hallucination.
The
reasoning-driven
term
captures
reasoning-driven instability that accumulates during autoregressive decoding. Here, we model gen-
eration as a martingale process, where deviation from the expectation is controlled by concentration
inequalities. Specifically, Freedman’s inequality (Geman et al., 1992) gives
∥uh −E[uh]∥≤K · exp

−Kϵ2
C

· α(eβT −1),
(6)
where K is the number of rollouts averaged, β summarizes per-step growth in local Jacobians, α
scales the cumulative effect and C is a task and model-dependent constant. This bound shows that
reasoning-driven hallucinations grow exponentially with sequence length T.
We now synthesize the two components into a unified result that characterizes the overall risk of
hallucination. By combining the NTK-conditioned approximation bound for data-driven deviation
with the Freedman-style concentration bound for reasoning-driven instability, we obtain the follow-
ing unified bound of data-driven and reasoning-driven hallucinations (detailed proof is provided in
Section A):
Theorem 3.2 (Hallucination Risk Bound).
Let u∗:= Φ(y∗) denote the semantic embed-
ding of the ground-truth output and uh := Φ(Y ) that of the model-generated output. Under
Assumptions A1-A3, suppose there exists β ≥0 such that

QT
t=1 Jt

2 ≤eβT . Then the total
hallucination risk satisfies
∥u∗−uh∥≤

1 + kpt log O(P, L) + k · ϵmismatch
Signalk

inf
u∈Uh ∥u∗−u∥
|
{z
}
data-driven term
+ |L| · exp

−Kϵ2
C

· α
 eβT −1

|
{z
}
reasoning-driven term
Here, |L| denotes the total sampled trajectories.
3.3
HALLUCINATION QUANTIFICATION VIA HALLUGUARD
While Theorem 3.2 makes explicit how data-driven and reasoning-driven hallucinations emerge and
evolve, applying it directly at inference is impractical since direct step-wise Jacobians for billion-
parameter LLMs are intractable, so we seek a proxy score that is computable, stable, and faithful to
our decomposition.
Let K denote the NTK Gram matrix with eigenvalues λ1 ≥· · · ≥λr > 0 and condition number
κ(K) = λmax/λmin. Let Jt be the step-t input-output Jacobian of the decoder, and define σmax :=
supt ∥Jt∥2 as the uniform spectral bound(note that σmax is independent of the spectrum of K).
Under Assumptions A1-A3, a standard NTK approximation argument yields infu∈Uh ∥u∗−u∥≤
Cd det(K)−cd ∥u∗∥, so that det(K) capture the representations in systematic bias.
5

## Page 6

Published as a conference paper at ICLR 2026
For autoregressive rollout, based on the property of Jacobian, we have
 QT
t=1 Jt

2
≤
QT
t=1 ∥Jt∥2
=
exp
 PT
t=1 log ∥Jt∥2

, so that we have
 QT
t=1 Jt

2
≤
eβT . Since
β ≤log σmax with σmax := supt ∥Jt∥2 thus we have the upper bound as ∥QT
t=1 Jt∥2 ≤σT
max =
e(log σmax)T . Thus, log σmax serves as a stable and tractable proxy for the per-step amplification rate.
Perturbation analysis of K, together with classical eigenvalue sensitivity results (Trefethen & Bau,
2022), yields Var[uh]
≤
cv κ(K)2 ∥δ∥2, showing that instability grows quadratically with the
condition number κ(K). To temper this effect and ensure additivity, we penalize ill-conditioned
representations via −log κ2, where log compression brings a well-behaved dynamic range.
Table 1: Correlation between NTK proxies
and task families.
SQuAD Math-500 TruthfulQA
det(K)
0.84
0.42
0.61
log σmax −log κ2
0.39
0.88
0.67
In summary, det(K) quantifies representational ad-
equacy, log σmax captures rollout amplification,
and −log κ2 penalizes spectral instability, together
forming a compact and tractable proxy consis-
tent with the Hallucination Risk Bound.
The
lightweight projection layers are self-supervised
spectral calibration modules, optimized offline (via
AdamW) to align NTK spectral properties across
heterogeneous backbones into a stable, comparable geometric space-without hallucination labels or
task-specific supervision, with the backbone fully frozen and zero runtime overhead during infer-
ence. Detailed proofs are provided in Section B.
Empirical validation.
We empirically validate how those proxies correlate with different task
families. In Table 1, det(K) correlates most strongly with the data-centric task SQuAD (0.84), in-
dicating its role in capturing factual fidelity. In contrast, for the reasoning-oriented MATH-500, the
highest correlation is observed with log σmax −log κ2 (0.88), reflecting the importance of amplifi-
cation and stability in multi-step reasoning.
Motivated by the above, we formally define HALLUGUARD as follows, which provides a principled
and unified lens for hallucination detection:
HALLUGUARD(uh) = det(K) + log σmax −log κ2.
(7)
4
EXPERIMENTS
We comprehensively evaluate HALLUGUARD across 10 diverse benchmarks, 11 competitive base-
lines, and 9 popular LLM backbones. We aim to evaluate its efficacy from the following five ques-
tions: Q1: How does HALLUGUARD perform across different task families? Q2: How does HAL-
LUGUARD perform across LLMs of different scales? Q3: How does each term capture trends across
task families? Q4: Can HALLUGUARD guide test-time inference to improve downstream reason-
ing? Q5: How well does HALLUGUARD generalize to detecting fine-grained hallucinations beyond
benchmarks?
Section 4.1 details the setup; Section 4.2 evaluates HALLUGUARD as a detection method(Q1–Q3),
Section 4.3 applies HALLUGUARD in score-guided inference(Q4) and Section 4.4 analyzes HAL-
LUGUARD on fine-grained hallucination via a case study on semantic data(Q5).
4.1
EVALUATION SETUP
Benchmarks. We evaluate across 10 widely used benchmarks spanning three distinct categories.
For data-grounded QA, we include RAGTruth (Niu et al., 2024), NQ-Open (Kwiatkowski et al.,
2019), HotpotQA (Yang et al., 2018) and SQuAD (Rajpurkar et al., 2016), which emphasize factual
correctness through external evidence. For reasoning-oriented tasks, we use GSM8K (Cobbe et al.,
2021), MATH-500 (Hendrycks et al., 2021), and BBH (Suzgun et al., 2023), which require multi-step
derivations prone to compounding errors. Finally, for instruction-following settings, we consider
TruthfulQA (Lin et al., 2022a), HaluEval (Li et al., 2023a) and Natural (Wang et al., 2022),
which probe hallucinations under open-ended or adversarial prompts.
6

## Page 7

Published as a conference paper at ICLR 2026
Baselines. We compare HALLUGUARD with 11 competitive detectors spanning diverse strategies.
Uncertainty-based methods include Perplexity (Ren et al., 2023), Length-Normalized Predictive
Entropy(LN-Entropy) (Malinin & Gales, 2021), Semantic Entropy (Kuhn et al., 2023), Energy
Score (Liu et al., 2020) and P(true) (Kadavath et al., 2022). Consistency-based approaches cover
SelfCheckGPT (Manakul et al., 2023), Lexical Similarity (Lin et al., 2022b), FActScore (Min et al.,
2023) and RACE (Wang et al., 2025). Internal-state methods are represented by Inside (Chen et al.,
2024) and MIND (Su et al., 2024).
LLM Backbone Models. We evaluate 9 publicly available LLMs spanning different scales and
architectures. These include five models from the Llama family (Llama2-7B, Llama2-13B, Llama2-
70B, Llama3-8B, and Llama3.2-3B) (Touvron et al., 2023; Grattafiori et al., 2024), along with OPT-
6.7B (Zhang et al., 2022), Mistral-7B-Instruct (Jiang et al., 2023), QwQ-32B (Yang et al., 2024),
and GPT-2 (117M) (Radford et al., 2019). All models are used in their off-the-shelf form with
pre-trained weights and tokenizers provided by Hugging Face, without further fine-tuning.
Evaluation Metrics. We evaluate hallucination detection ability under two regimes following Ja-
niak et al. (2025): ROUGE-based reference evaluation (∗r) and LLM-AS-A-JUDGE (∗llm). For
performance measures, we report the area under the receiver operating characteristic curve (AU-
ROC) and the area under the precision-recall curve (AUPRC). AUROC is widely used to assess the
quality of binary classifiers and uncertainty estimators, while AUPRC highlights performance under
class imbalance. In both cases, higher values indicate better detection.
4.2
MAIN RESULTS
Q1: How does HALLUGUARD perform across different task families? To evaluate how HAL-
LUGUARD performs across different task types, we conduct experiments on all benchmarks. For
clarity, Table 2 presents representative results from three task families: data-centric (RAGTruth),
reasoning-oriented (Math-500), and instruction-following (TruthfulQA). As shown, HAL-
LUGUARD consistently outperforms all baselines across backbones. On Math-500, it reaches
81.76% AUROC and 79.76% AUPRC, improving over the second-best method by up to 8.3%.
On RAGTruth, it attains 84.59% AUROC and 81.15% AUPRC, with gains of up to 7.7%. On
TruthfulQA, it achieves 77.05% AUROC and 73.79% AUPRC, exceeding the next strongest base-
line by as much as 6.2%. Overall, HALLUGUARD establishes new state-of-the-art results across di-
verse task families, with particularly pronounced improvements on reasoning-oriented benchmarks.
Q2:
How does HALLUGUARD perform across LLMs of different scales?
We fur-
ther investigate whether the effectiveness of HALLUGUARD depends on model scale, as
smaller backbones are typically more prone to hallucination.
Table 3 reports representa-
tive results on small(Llama2-7B, Llama3-8B), mid-sized(Llama2-13B), and large-scale(Llama2-
70B) models using SQuAD, GSM8K, and HaluEval.
Across all settings, HALLUGUARD
consistently surpasses baselines, with the largest margins on smaller models-for instance,
Figure 2:
Ablation results comparing in-
dividual terms with ground-truth trends on
SQuAD (top) and Math-500 (bottom).
72.89% AUPRCr on HaluEval with Llama2-7B,
more than 10% above the second best.
Mid-
sized models also exhibit clear gains (e.g., 79.01%
AUROCr on GSM8K), while even large-scale mod-
els like Llama2-70B see steady improvements (e.g.,
83.8% AUROCr on SQuAD). Overall, HALLU-
GUARD benefits most on small backbones while
maintaining consistent advantages across scales.
Q3: How does each term capture trends across
task families?
As shown in Figure 2, each term
faithfully tracks the ground-truth trend within its re-
spective task family. On data-centric SQuAD, the
data-driven term closely follows the dashed gold
curve across the variant hallucination rate, capturing
the smooth AUROC decline. On reasoning-oriented
MATH-500, the reasoning-driven term mirrors the
monotonic AUROC drop as reasoning drift in-
7

## Page 8

Published as a conference paper at ICLR 2026
Table 2:
Performance comparison on representative benchmarks:
data-centric (RAGTruth),
reasoning-oriented (Math-500), and instruction-following (TruthfulQA). We highlight the first
and second best results.
GPT2
OPT-6.7B
Mistral-7B
QwQ-32B
AUROCr
AUPRCr
AUROCllm
AUPRCllm
AUROCr
AUPRCr
AUROCllm
AUPRCllm
AUROCr
AUPRCr
AUROCllm
AUPRCllm
AUROCr
AUPRCr
AUROCllm
AUPRCllm
RAGTruth
HALLUGUARD 75.51 73.40 62.40 56.60
80.13 76.77 71.01 63.58
82.31 80.79 64.89 67.25
84.59 81.15 71.82 66.68
Inside
73.42 73.08 61.99 56.39
79.49 71.82 66.1 62.46
75.32 73.19 64.58 61.05
77.72 73.47 66.05 64.73
MIND
58.54 54.79 43.47 41.85
63.82 62.58 51.03 44.78
73.13 71.53 58.25 58.6
64.23 63.06 47.37 51.47
Perplexity
58.07 56.68 43.84 41.53
64.47 61.57 47.12 52.98
65.42 63.63 53.28 51.36
73.91 72.92 60.81 59.77
LN-Entropy
64.42 60.79 49.41 45.04
60.81 57.91 48.76 42.27
64.22 60.92 52.24 48.41
63.81 62.26 47.52 52.17
Energy
65.53 62.42 51.8 47.22
66.54 63.28 54.21 49.19
64.36 62.26 48.64 53.93
73.26 71.21 65.43 62.32
Semantic Ent.
60.72 59.41 50.55 45.86
70.2 68.34 54.54 56.74
66.01 64.49 53.01 55.5
66.48 64.41 51.54 50.11
Lexical Sim.
64.72 63.1 55.04 48.04
67.28 64.62 52.55 54.86
64.96 61.17 52.34 45.11
70.87 67.41 61.25 51.01
SelfCheckGPT
65.4 62.79 52.85 52.43
66.64 64.89 52.69 51.17
71.19 68.45 63.13 60.23
65.79 62.45 54.76 51.29
RACE
64.83 62.84 51.8 48.44
64.26 61.03 52.74 46.22
66.34 64.54 51.88 53.86
71.13 69.96 57.58 55.54
P(true)
66.19 64.04 48.2 56.27
68.44 65.48 57.53 53.08
72.54 71.8 57.25 59.42
65.32 63.01 53.01 52.32
FActScore
65.72 64.39 51.94 47.51
61.53 58.2 51.86 45.57
63.98 60.71 53.54 49.34
66.72 64.03 58.21 49.17
BBH
HALLUGUARD 71.06 67.94 62.05 59.05
73.1 70.88 63.67 61.88
79.85 76.5 67.13 60.57
81.76 79.76 68.77 65.46
Inside
66.18 66.81 56.15 58.62
70.64 65.22 63.28 59.28
67.2 65.49 51.3 53.46
80.8 71.49 64.05 63.42
MIND
55.41 51.77 39.01 41.59
55.48 53.46 38.59 40.88
65.71 63.7 49.61 52.54
61.75 60.18 53.46 50.04
Perplexity
53.28 50.22 43.86 38.98
64.89 62.12 48.65 51.99
61.97 60.05 51.15 42.87
60.28 57.75 51.62 43.38
LN-Entropy
60.84 58.76 42.76 47.48
58.71 55.01 43.55 42.02
68.96 69.44 58.79 57.49
63.96 62.18 46.01
49.5
Energy
55.09 51.99 46.2
39.5
53.96 50.98 42.56 34.12
66.27 62.72 49.48 50.06
69.61 68.66 54.35 57.36
Semantic Ent.
58.16 54.81 49.61 40.39
62.63 59.52 50.14 45.02
64.99 61.33 50.11 45.53
62.76 60.95 45.77 45.75
Lexical Sim.
51.37 47.18 38.37 39.06
61.27 58.06 44.13 42.96
58.25 55.92 46.31 46.01
69.46 67.59 55.93
52.6
SelfCheckGPT
54.51 51.86 44.62 44.01
57.36 53.21 42.55 38.27
63.68 62.5
51.7 53.03
64.56 62.49 55.85
45.8
RACE
55.99 54.66 41.39 38.32
64.23 62.03 56.03 53.44
66.88 64.33 49.57 48.5
59.5 55.83 46.13 41.07
P(true)
54.57 52.88 45.45 44.74
57.02 55.49 48.81 37.84
57.11 55.21 43.93 47.05
61.49 59.03 44.37 44.69
FActScore
56.76 53.85 40.25 40.01
54.51 53.2 38.45 36.49
62.11 58.64 53.52 47.27
58.82 57.47 49.48 42.74
TruthfulQA
HALLUGUARD 72.1 68.76 60.09 52.01
69.59 68.36 58.52 52.65
77.05 73.79 63.62 62.26
74.26 72.76 57.39 64.07
Inside
70.42 68.76 60.09 52.01
62.1 59.78 51.07 51.38
62.53 60.99 52.3 49.35
70.89 64.44 56.61 56.01
MIND
59.45 56.79 45.22 43.71
60.56 58.55 47.49 49.63
59.2 57.98 47.23 41.79
62.81 61.5 52.56 46.37
Perplexity
50.57 47.87 40.64 35.63
55.07 52.26 44.43 42.79
60.8 59.69 47.33 41.62
55.29 52.46 43.95 43.92
LN-Entropy
58.04 56.99 41.94 47.21
56.12 54.01 47.06 38.4
59.67 56.25 41.99 41.25
60.76 58.21 46.24 42.64
Energy
55.02 53.31 38.78 45.16
54.42 51.85 36.21 42.57
58.93 55.25 50.76 41.72
64.15 61.32 51.78 50.02
Semantic Ent.
61.01 57.08 43.35 45.2
51.48 47.81 34.15 38.16
54.44 53.33 36.62 40.35
66.75 63.85 51.11 46.71
Lexical Sim.
52.54 50.56 39.94 33.42
59.74 55.72 49.89 46.81
66.16 64.05 54.08 51.65
55.24 51.36 46.39 39.57
SelfCheckGPT
56.04 54.48 43.78 44.38
58.93 56.47 47.65 39.02
61.14 58.91 42.97 47.01
55.86 54.95 41.08 37.35
RACE
53.02 50.33 41.7 33.81
62.95 67.89 54.61 51.93
71.06 68.49 60.4 57.44
55.75 52.62 46.5
43.19
P(true)
55.52 53.41 38.33 38.38
54.88 53.1 38.22 40.96
55.8 52.01 40.88 38.72
57.18 55.16 46.19 38.21
FActScore
53.82 51.42 41.33 35.2
54.57 51.26 42.51 35.52
53.97 50.2 42.97 36.16
62.31 60.23 45.06
49.9
creases. These results show that each term is well
matched to its task family and faithfully tracks performance trends as hallucination rates rise.
4.3
TEST-TIME INFERENCE
Test-time reasoning remains challenging, as models need to generate coherent multi-step solu-
tions without drifting into errors. To assess whether hallucination detection can mitigate this dif-
ficulty, we integrate detectors into beam search and evaluate Qwen2.5-Math-7B on MATH-500 and
Llama3.1-8B on Natural. As shown in Table 4, HALLUGUARD achieves the strongest gains: on
MATH-500, it reaches 81.00% accuracy, around 10% higher than IO Prompt; on Natural, it at-
tains 70.96%, exceeding IO Prompt by 15.72%. These results demonstrate that HALLUGUARD not
only detects hallucinations but also strengthens test-time reasoning by guiding models toward more
reliable solutions.
4.4
CASE STUDY
Fine-grained hallucinations-lexically similar yet semantically incorrect outputs-pose a particular
challenge for detection. To evaluate whether HALLUGUARD can comprehensively capture such
subtle errors, we use the PAWS dataset (Zhang et al., 2019), which contrasts paraphrases with high
surface overlap but divergent meanings. Following Li et al. (2025), we adopt ROUGE-based refer-
ence signals for evaluation (Table 5). Across model scales, HALLUGUARD consistently surpasses
8

## Page 9

Published as a conference paper at ICLR 2026
Table 3: Performance comparison across backbone scales (small, mid-sized, and large) on three
benchmarks: SQuAD, GSM8K, HaluEval. We highlight the first and second best results.
Llama2-7B
Llama-3-8B
Llama2-13B
Llama2-70B
AUROCr
AUPRCr
AUROCllm
AUPRCllm
AUROCr
AUPRCr
AUROCllm
AUPRCllm
AUROCr
AUPRCr
AUROCllm
AUPRCllm
AUROCr
AUPRCr
AUROCllm
AUPRCllm
SQuAD
HALLUGUARD 81.05 77.16 71.18 64.38
79.56 78.29 67.97 63.27
81.45 78.39 64.39 65.07
83.8 81.77 70.46 73.24
Inside
73.63 75.74 65.22 59.11
76.13 72.44 65.62 62.94
74.68 74.81 61.01 59.51
81.24 75.09 69.48
62.4
MIND
64.57 61.11 52.39 53.13
62.29 59.58 44.49 48.61
68.64 66.95 54.92 52.49
73.46 71.71 57.76 56.77
Perplexity
63.93 61.77 46.97 48.2
70.51 67.51 55.71 52,68
70.19 69.22 60.33 54.82
74.23 70.88 62.24 58.05
LN-Entropy
65.96 64.22 53.43 52.84
63.7
60.4 46.19 42.85
61.66 59.16 49.05 46.27
72.44 68.91 56.77 52.63
Energy
59.83 56.11 46.19 43.18
64.41 61.02 56.17 46.21
61.02 59.73 48.26 42.08
69.01 66.19 58.44 49.82
Semantic Ent.
60.29 57.73 43.63 48.83
66.52 62.62 52.37 52.7
70.58 67.22 53.31 52.94
72.01 68.51 56.49
50.9
Lexical Sim.
70.31 69.08 53.97 53.31
66.43 63.56 53.19 50.96
68.53 67.42 50.73 54.12
68.95 67.91 60.52 56.56
SelfCheckGPT
68.26 67.09 60.06 57.31
73.99 72.15 65.26 54.02
65.47 61.65 53.12 49.89
73.07 70.49 56.59 54.65
RACE
71.35 69.23 59.18 54.73
68.17 66.02 54.65 53.06
64.19 60.45 47.53 45.66
64.05 62.39 54.38 50.07
P(true)
62.55 61.09 46.84 52.32
67.42 63.94 55.35 47.52
71.56 68.4 57.51 45.66
66.81 62.71 57.43 46.85
FActScore
70.32 68.63 58.13 53.01
71.2 69.45 61.92 54.91
66.65 63.2 56.41 53.42
68.33 65.26 56.93 48.46
GSM8K
HALLUGUARD 75.89 72.83 62.29 63.46
75.2
72.9 63.62 61.79
79.01 76.73 64.38 64.97
77.33 73.97 60.48 61.26
Inside
74.61 68.35 58.57 62.58
73.73 67.51 56.02 57.28
75.79 76.26 60.91 59.77
72.3 72.26 54.49 58.39
MIND
65.88 63.4 48.28 48.17
66.57 65.55 48.84 53.4
61.49 59.55 51.63 51.45
66.41 63.44 52.05 53.57
Perplexity
66.23 64.1 53.52 52.31
57.61 53.63 41.37 41.59
60.96 58.67 46.27 47.44
64.32 62.81 51.15
51.3
LN-Entropy
59.45 55.95 43.04 44.08
68.22 66.05 53.03 53.21
61.31 58.90 45.83 40.86
61.81 60.46 44.5
44.76
Energy
58.15 54.71 43.65 36.71
59.79 56.52 50.31 42.23
57.58 56.07 43.39 38.94
65.27 62.94 52.8
46.6
Semantic Ent.
57.95 54.68 42.78 41.95
66.9 64.81 50.47 55.36
62.72 59.09 49.33 44.35
60.63 57.01 46.22 40.24
Lexical Sim.
65.8
63.7 52.12 54.07
63.29 59.87 53.17 50.02
63.83 60.20 54.43 44.82
63.27 59.41 47.42 47.38
SelfCheckGPT
60.99 57.54 49.28 44.43
65.72 62.01 54.49 50.34
57.98 54.58 46.72 39.86
68.06 65.09 52.99 50.89
RACE
63.37 62.33 53.53 49.94
64.49 61.47 53.28 47.55
64.20 61.96 50.15 45.35
68.35 66.66 50.41 51.16
P(true)
65.95 63.63 54.95 48.25
62.59 58.88 47.21 42.2
67.08 65.60 53.66 55.12
60.16 58.14 47.73 49.49
FActScore
56.69 53.71 45.78 39.52
65.69 61.95 53.69 46.06
55.76 54.17 44.91 43.18
59.84 55.85 44.05 39.49
HaluEval
HALLUGUARD 75.72 72.89 66.65 63.15
73.43 71.19 64.95 54.8
78.15 74.15 65.39 61.14
80.79 79.54 67.68 68.51
Inside
71.33 67.63 59.73 53.15
67.95 64.93 60.31 52.21
72.01 71.97 56.51 60.64
74.62 68.33 62.22
64.4
MIND
54.8 51.43 44.15 43.34
64.54 60.89 49.09 45.13
55.05 53.28 39.16 45.17
57.98 56.01 45.82 41.69
Perplexity
54.02 52.53 38.76 40.51
61.31 59.36 50.62 46.01
54.99 51.39 42.64 35.64
62.85 60.59 48.29 43.85
LN-Entropy
59.47 58.33 50.2 46.91
64.89 60.72 51.78 46.39
65.18 63.53 49.70 48.09
60.16 58.89 50.29 48.42
Energy
62.29 59.6 50.68 42.24
62.74 61.61 50.17 52.01
60.54 59.04 43.53 50.37
60.13 58.44 48.79 48.01
Semantic Ent.
59.39 55.94 48.53 46.35
55.25 53.05 44.5 44.35
59.44 57.72 45.38 40.77
61.57 57.99 49.07 45.39
Lexical Sim.
63.61 61.16 55.01 44.75
56.59 55.39 44.45 45.57
53.46 52.06 41.34 40.57
64.37 60.92 54.29 50.86
SelfCheckGPT
64.29 61.83 48.4 45.49
65.44 63.13 57.02 48.23
65.24 63.52 53.71 54.33
57.12 55.26 40.5
43.06
RACE
59.78 59.14 48.1 40.47
61.98 60.32 48.08 46.29
60.65 59.11 49.92 44.51
62.11 58.24 40.5
43.06
P(true)
57.46 54.8 41.84 40.47
56.32 54.04 42.55 43.75
65.77 63.01 49.98 45.47
55.75 54.94 44.14 43.97
FActScore
63.93 61.33 46.9 51.87
61.73 57.85 49.92 42.15
65.15 63.71 55.98 54.61
62.66 60.3 53.13 46.42
Table 4: Performance of hallucination score-guided test-time inference across reasoning tasks. We
highlight the first and second best results.
Dataset
IO
Prompt
Ours
Inside
MIND
Perplexity
LN-
Entropy
Energy
Semantic
Ent.
SelfCheck-
GPT
RACE
P(true)
FActScore
MATH-500
72.70
81.00
74.90
77.10
77.10
76.20
78.00
72.50
74.00
75.10
67.10
71.60
Natural
55.24
70.96
67.42
68.32
67.51
68.04
68.59
68.10
65.68
66.90
68.16
67.74
baselines: it achieves 90.18% AUROC and 87.64% AUPRC on Llama2-70B, and 91.24% AUROC
and 88.53% AUPRC on QwQ-32B-exceeding the next-best method by nearly five points. Even on
GPT-2, it leads with 83.27% AUROC and 80.46% AUPRC. These results confirm HALLUGUARD’s
effectiveness in capturing fine-grained semantic inconsistencies beyond benchmark settings.
Table 5: Results on PAWS measuring semantic hallucination detection with Llama-3.2-3B, Llama2-
70B, and QwQ-32B. We highlight the first and second best results.
Method
Ours
Inside
MIND
Perplexity
LN-
Entropy
Energy
Semantic
Ent.
Lexical
Sim.
SelfCheck-
GPT
RACE
P(true)
FActScore
Llama3.2 AUROC
85.63
80.46
78.93
71.27
72.19
73.05
75.11
64.58
77.82
79.47
73.56
68.44
AUPRC
82.14
77.28
75.41
67.55
68.34
70.22
72.41
59.67
73.41
76.28
70.43
63.58
Llama2
AUROC
90.18
85.47
83.92
75.68
76.23
77.14
79.06
68.35
82.71
84.26
77.39
72.62
AUPRC
87.64
82.38
81.06
71.42
72.59
74.28
76.32
63.44
78.89
81.73
74.18
67.58
QwQ
AUROC
91.24
85.41
84.56
76.72
77.43
78.29
80.42
69.54
83.59
86.38
78.53
73.46
AUPRC
88.53
82.27
81.37
72.63
73.29
75.44
77.18
64.27
79.42
83.41
75.21
68.32
9

## Page 10

Published as a conference paper at ICLR 2026
5
RELATED WORK
In this section, we review prior hallucination-detection methods by their detection target–Data-
driven hallucinations and reasoning-driven hallucinations.
Detecting Data-Driven Hallucinations. Recent work has shown that internal activations encode
rich indicators of such flaws. Chen et al. (2024) proposed EIGENSCORE, which computes statis-
tics of hidden representations from the eigen matrix to estimate hallucination risk. Su et al. (2024)
introduced MIND, an unsupervised detector that models temporal dynamics of hidden states with-
out requiring labels, along with HELM benchmark to enable standardized evaluation. Azaria &
Mitchell (2023) demonstrated using linear probes on intermediate states to predict truthfulness.
Detecting Reasoning-Driven Hallucinations. There are other works targeting inference-time in-
consistencies during generation-such as logical errors, instability across decoding steps, or tempo-
ral drift in extended outputs. Manakul et al. (2023) proposed SELFCHECKGPT, which assesses
self-consistency by sampling multiple candidate generations and measuring their alignment using
entailment and lexical overlap. Kalai & Vempala (2024) introduced a suite of calibration-based un-
certainty scores designed to capture hallucination risk directly from output distributions. Ding et al.
(2025) proposed REACTSCORE, which integrates entropy with intermediate reasoning traces to de-
tect failures in multi-step decision-making. FACTSCORE (Min et al., 2023) decomposes outputs into
atomic factual units and verifies each against retrieved passages using entailment-based scoring.
6
CONCLUSION
The reliability of LLMs is often undermined by hallucinations, which arise from two main sources:
data-driven, caused by flawed knowledge acquired during training, and reasoning-driven, stemming
from inference-time instabilities in multi-step generation. Although these hallucinations frequently
evolve in practice, existing detectors usually target only one source and lack a solid theoretical
foundation. To address this gap, we propose a unified theoretical framework–a Hallucination Risk
Bound, which formally decomposes hallucination risk into data-driven and reasoning-driven com-
ponents, offering a principled view of how hallucinations emerge and evolve during generation.
Building on this foundation, we introduce HALLUGUARD, a NTK-based score that measures sensi-
tivity to semantic perturbations and captures internal instabilities, thereby enabling holistic detection
of both data-driven and reasoning-driven hallucinations. We evaluate HALLUGUARD across 10 di-
verse benchmarks, 11 competitive baselines, and 9 popular LLM backbones, where it consistently
achieves state-of-the-art performance, demonstrating robustness and practical efficacy. Looking
forward, leveraging HalluGuard’s sensitivity to error propagation offers a promising pathway for
developing prognostic indicators in interactive multi-turn dialogues, enabling systems to predict and
preempt hallucinations before they fully manifest.
REPRODUCIBILITY STATEMENT
We have taken several measures to ensure the reproducibility of our work. A complete description
of the theoretical framework, including the formal assumptions and proofs of the Hallucination Risk
Bound, is provided in Section 3 and Section A. Detailed experimental settings and evaluation pro-
tocols are documented in Section 4 and Section C.1, covering all 10 benchmarks, 11 baselines, and
9 LLM backbones. Together, these resources ensure that both our theoretical claims and empirical
results can be independently validated and extended by the community.
ETHICS STATEMENT
This study is based exclusively on publicly available datasets and open-source large language mod-
els, and does not involve human subjects or the use of private data. All scientific concepts, method-
ological designs, experimental implementations, and resulting conclusions remain entirely the re-
sponsibility of the authors.
10

## Page 11

Published as a conference paper at ICLR 2026
ACKNOWLEDGEMENTS
We thank the anonymous reviewers for their constructive comments. This work is supported by
the National Science Foundation under Award No. IIS-2339989 and No. 2406439, DARPA under
contract No. HR00112490370 and No. HR001124S0013, U.S. Department of Homeland Security
under Grant Award No. 17STCIN00001-08-00, Amazon-Virginia Tech Initiative for Efficient and
Robust Machine Learning, Amazon AWS, Google, Cisco, 4-VA, Commonwealth Cyber Initiative,
National Surface Transportation Safety Center for Excellence, and Virginia Tech. The views and
conclusions are those of the authors and should not be interpreted as representing the official policies
of the funding agencies or the government.
REFERENCES
Amos Azaria and Tom Mitchell. The internal state of an llm knows when it’s lying. In Findings
of the Association for Computational Linguistics: Conference on Empirical Methods in Natural
Language Processing(EMNLP), 2023.
Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx,
Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson,
Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel,
Jared Quincy Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Ste-
fano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren
Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Pe-
ter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard,
Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte
Khani, Omar Khattab, Pang Wei Koh, Mark Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya
Kumar, Faisal Ladhak, Mina Lee, Tony Lee, Jure Leskovec, Isabelle Levent, Xiang Lisa Li,
Xuechen Li, Ece Kamar, Michal Kosinski, Ryan Chi-Ying Hsieh, Drew A. Linsley, Long O. Mai,
Nikolay Manchev, Christopher D. Manning, Yian Yin, Christopher J. N. de M. L. Matthews, Lu-
cia Mondragon, Ognjen Oreskovic, Mark Sabini, Yusuf Sahin, Clark Barrett, Christopher Potts,
James Y. Zou, Jiajun Wu, and Percy Liang. On the opportunities and risks of foundation models.
ArXiv, 2021. URL https://crfm.stanford.edu/assets/report.pdf.
Jean C´ea. Approximation variationnelle des probl`emes aux limites. In Annales de l’institut Fourier,
volume 14, pp. 345–444, 1964.
Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. Inside:
Llms’ internal states retain the power of hallucination detection. International Conference on
Learning Representations(ICLR), 2024.
Yuyan Chen, Qiang Fu, Yichen Yuan, Zhihao Wen, Ge Fan, Dayiheng Liu, Dongmei Zhang, Zhixu
Li, and Yanghua Xiao. Hallucination detection: Robustly discerning reliable answers in large
language models. Conference on Information and Knowledge Management(CIKM), 2023.
Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming.
Conference on Neural Information Processing Systems(NeurIPS), 2019.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John
Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.
org/abs/2110.14168.
Fabio Dennst¨adt, Janna Hastings, Paul Martin Putora, Max Schmerder, and Nikola Cihoric. Im-
plementing large language models in healthcare while balancing control, collaboration, costs and
security. NPJ digital medicine, 8(1):143, 2025.
Yue Ding, Xiaofang Zhu, Tianze Xia, Junfei Wu, Xinlong Chen, Qiang Liu, and Liang Wang.
D2hscore: Reasoning-aware hallucination detection via semantic breadth and depth analysis in
llms, 2025. URL https://arxiv.org/abs/2509.11569.
Stuart Geman, Elie Bienenstock, and Ren´e Doursat. Neural networks and the bias/variance dilemma.
Neural computation, 4(1):1–58, 1992.
11

## Page 12

Published as a conference paper at ICLR 2026
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan,
Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Ko-
renev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava
Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux,
Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret,
Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius,
Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary,
Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab
AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco
Guzm´an, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind That-
tai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Kore-
vaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra,
Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Ma-
hadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu,
Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jong-
soo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala,
Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid
El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren
Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin,
Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi,
Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew
Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Ku-
mar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoy-
chev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur C¸ elebi, Patrick Alrassy, Pengchuan
Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan,
Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ra-
mon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Ro-
hit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan
Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell,
Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng
Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer
Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman,
Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mi-
haylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor
Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, V´ıtor Albiero, Vladan Petrovic, Weiwei
Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang
Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Gold-
schlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning
Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh,
Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria,
Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein,
Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, An-
drew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, An-
nie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel,
Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leon-
hardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu
Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Mon-
talvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao
Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia
Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide
Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le,
Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily
Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smoth-
ers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni,
Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia
Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan,
Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harri-
son Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj,
12

## Page 13

Published as a conference paper at ICLR 2026
Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James
Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jen-
nifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang,
Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Jun-
jie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy
Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang,
Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell,
Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa,
Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias
Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L.
Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike
Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari,
Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan
Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong,
Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent,
Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar,
Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Ro-
driguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy,
Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin
Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon,
Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ra-
maswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha,
Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal,
Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satter-
field, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj
Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo
Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook
Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Ku-
mar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov,
Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiao-
jian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia,
Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao,
Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhao-
duo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL
https://arxiv.org/abs/2407.21783.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. Confer-
ence on Neural Information Processing Systems(NeurIPS), 2021.
Lei Huang, Weijiang Yu, Weitao Wang, Yujia Wang, Shi-Qi Chen, and Ju-Hua Wang. A survey
on hallucination in large language models: Principles, taxonomy, challenges, and open questions.
ACM Transactions on Information Systems, 2025.
Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for
long document summarization. In The North American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies(NAACL), pp. 1419–1436, Online, June 2021.
Association for Computational Linguistics(ACL). doi: 10.18653/v1/2021.naacl-main.112. URL
https://aclanthology.org/2021.naacl-main.112.
Arthur Jacot, Franck Gabriel, and Cl´ement Hongler. Neural tangent kernel: Convergence and gen-
eralization in neural networks. Conference on Neural Information Processing Systems(NeurIPS),
2018.
Denis Janiak, Jakub Binkowski, Albert Sawczyn, Bogdan Gabrys, Ravid Shwartz-Ziv, and Tomasz
Kajdanowicz. The illusion of progress: Re-evaluating hallucination detection in llms. Conference
on Empirical Methods in Natural Language Processing(EMNLP), 2025.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM
13

## Page 14

Published as a conference paper at ICLR 2026
Computing Surveys, 55(12):1–38, March 2023. ISSN 1557-7341. doi: 10.1145/3571730. URL
http://dx.doi.org/10.1145/3571730.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chap-
lot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier,
L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril,
Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. URL https:
//arxiv.org/abs/2310.06825.
Peizhong Ju, Xiaojun Lin, and Ness B. Shroff.
On the generalization power of the overfitted
three-layer neural tangent kernel model.
Conference on Neural Information Processing Sys-
tems(NeurIPS), 2022.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez,
Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer
El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bow-
man, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna
Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom
Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Ka-
plan. Language models (mostly) know what they know, 2022.
Adam Tauman Kalai and Santosh S. Vempala. Calibrated language models must hallucinate. ACM
Symposium on Theory of Computing (STOC), 2024.
Markus Kattnig, Alessa Angerschmid, Thomas Reichel, and Roman Kern. Assessing trustworthy ai:
Technical and legal perspectives of fairness in ai. Computer Law & Security Review, 55:106053,
2024.
Zong Ke, Yuqing Cao, Zhenrui Chen, Yuchen Yin, Shouchao He, and Yu Cheng. Early warning of
cryptocurrency reversal risks via multi-source data. Finance Research Letters, pp. 107890, 2025.
Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances
for uncertainty estimation in natural language generation. International Conference on Learning
Representations(ICLR), 2023.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research. Transactions of the
Association for Computational Linguistics(TACL), 7:452–466, 2019. doi: 10.1162/tacl a 00276.
URL https://aclanthology.org/Q19-1026/.
Jaehoon Lee, Samuel S. Schoenholz, Jeffrey Pennington, Ben Adlam, Lechao Xiao, Roman Novak,
and Jascha Sohl-Dickstein. Finite versus infinite neural networks: an empirical study. Conference
on Neural Information Processing Systems(NeurIPS), 2020a.
Jaehoon Lee, Lechao Xiao, Samuel S Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-
Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models
under gradient descent *. Journal of Statistical Mechanics: Theory and Experiment, 2020(12):
124002, December 2020b. ISSN 1742-5468. doi: 10.1088/1742-5468/abc62b. URL http:
//dx.doi.org/10.1088/1742-5468/abc62b.
Jiawei Li, Akshayaa Magesh, and Venugopal V. Veeravalli. Principled detection of hallucinations in
large language models via multiple testing, 2025. URL https://arxiv.org/abs/2508.
18473.
Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A large-
scale hallucination evaluation benchmark for large language models. Conference on Empirical
Methods in Natural Language Processing(EMNLP), 2023a.
Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating ob-
ject hallucination in large vision-language models. In Conference on Empirical Methods in Natu-
ral Language Processing(EMNLP), 2023b. URL https://openreview.net/forum?id=
xozJw0kZXF.
14

## Page 15

Published as a conference paper at ICLR 2026
Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization
branches out, pp. 74–81, 2004.
Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human
falsehoods. Association for Computational Linguistics(ACL), 2022a.
Zi Lin, Jeremiah Zhe Liu, and Jingbo Shang. Towards collaborative neural-symbolic graph se-
mantic parsing via uncertainty. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio
(eds.), Findings of the Association for Computational Linguistics(ACL), pp. 4160–4173, Dublin,
Ireland, May 2022b. Association for Computational Linguistics(ACL). doi: 10.18653/v1/2022.
findings-acl.328. URL https://aclanthology.org/2022.findings-acl.328/.
Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou,
and Sheng Liu. More thinking, less seeing? assessing amplified hallucination in multimodal
reasoning models. Conference on Neural Information Processing Systems(NeurIPS), 2025.
Weitang Liu, Xiaoyun Wang, John D. Owens, and Yixuan Li. Energy-based out-of-distribution
detection. Conference on Neural Information Processing Systems(NeurIPS), 2020.
Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction.
International Conference on Learning Representations(ICLR), 2021.
Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. Selfcheckgpt: Zero-resource black-box
hallucination detection for generative large language models. Conference on Empirical Methods
in Natural Language Processing(EMNLP), 2023.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen tau Yih, Pang Wei Koh, Mohit Iyyer,
Luke Zettlemoyer, and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of factual
precision in long form text generation. Conference on Empirical Methods in Natural Language
Processing(EMNLP), 2023.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song, and
Tong Zhang. Ragtruth: A hallucination corpus for developing trustworthy retrieval-augmented
language models. Association for Computational Linguistics(ACL), 2024.
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language
models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions
for machine comprehension of text. Conference on Empirical Methods in Natural Language
Processing(EMNLP), 2016.
Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan,
and Peter J. Liu. Out-of-distribution detection and selective generation for conditional language
models. International Conference on Learning Representations(ICLR), 2023.
Tom´aˇs Koˇ cisk´y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, G´abor Melis,
and Edward Grefenstette. The NarrativeQA reading comprehension challenge. Transactions of
the Association for Computational Linguistics(TACL), 2018.
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation
reduces hallucination in conversation. In Conference on Empirical Methods in Natural Language
Processing(EMNLP), 2021.
Weihang Su, Changyue Wang, Qingyao Ai, Yiran HU, Zhijing Wu, Yujia Zhou, and Yiqun Liu. Un-
supervised real-time hallucination detection based on the internal states of large language models.
Association for Computational Linguistics(ACL), 2024.
Zhongxiang Sun, Qipeng Wang, Haoyu Wang, Xiao Zhang, and Jun Xu. Detection and mitigation
of hallucination in large reasoning models: A mechanistic perspective. Conference on Neural
Information Processing Systems(NeurIPS) Workshop, 2025.
15

## Page 16

Published as a conference paper at ICLR 2026
Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung,
Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging big-
bench tasks and whether chain-of-thought can solve them. Findings of the Association for Com-
putational Linguistics(ACL), 2023.
Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kavya Elangovan, Lio Gutierrez, Teng Fong
Tan, and Daniel Shu Wei Ting. Large language models in medicine. Nature Medicine, 29(8):
1930–1940, 2023.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Niko-
lay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher,
Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy
Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn,
Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel
Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee,
Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra,
Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi,
Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh
Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen
Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic,
Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models,
2023. URL https://arxiv.org/abs/2307.09288.
Lloyd N Trefethen and David Bau. Numerical linear algebra. Society for Industrial and Applied
Mathematics(SIAM), 2022.
Roman Vershynin. High-dimensional probability: An introduction with applications in data science,
volume 47. Cambridge university press, 2018.
Changyue Wang, Weihang Su, Qingyao Ai, and Yiqun Liu. Joint evaluation of answer and reasoning
consistency for hallucination detection in large reasoning models, 2025.
Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, An-
jana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan
Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson,
Kirby Kuznia, Krima Doshi, Maitreya Patel, Kuntal Kumar Pal, Mehrad Moradshahi, Mihir Par-
mar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri,
Rushang Karia, Shailaja Keyur Sampat, Savan Doshi, Siddhartha Mishra, Sujan Reddy, Sumanta
Patro, Tanay Dixit, Xudong Shen, Chitta Baral, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi,
and Daniel Khashabi. Super-naturalinstructions: Generalization via declarative instructions on
1600+ nlp tasks. Conference on Empirical Methods in Natural Language Processing(EMNLP),
2022.
Zeyu Wei, Shuo Wang, Xiaohui Rong, Xuemin Liu, and He Li. Shadows in the attention: Contextual
perturbation and representation drift in the dynamics of hallucination in llms, 2025. URL https:
//arxiv.org/abs/2505.16894.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li,
Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang,
Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jin-
gren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin
Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao,
Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wen-
bin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng
Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu,
Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. URL
https://arxiv.org/abs/2407.10671.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. Conference on Empirical Methods in Natural Language Processing(EMNLP), 2018.
16

## Page 17

Published as a conference paper at ICLR 2026
Xinyue Zeng, Haohui Wang, Junhong Lin, Jun Wu, Tyler Cody, and Dawei Zhou. Lensllm: Un-
veiling fine-tuning dynamics for llm selection.
International Conference on Machine Learn-
ing(ICML), 2025.
Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A. Smith. How language model
hallucinations can snowball. International Conference on Machine Learning(ICML), 2023.
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christo-
pher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer,
Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettle-
moyer. Opt: Open pre-trained transformer language models, 2022. URL https://arxiv.
org/abs/2205.01068.
Yuan Zhang, Jason Baldridge, and Luheng He. Paws: Paraphrase adversaries from word scrambling.
The North American Chapter of the Association for Computational Linguistics(NAACL), 2019.
Zhenliang Zhang, Xinyu Hu, Huixuan Zhang, Junzhe Zhang, and Xiaojun Wan. Icr probe: Tracking
hidden state dynamics for reliable hallucination detection in llms. Association for Computational
Linguistics(ACL), 2025.
Weihong Zhong, Xiaocheng Feng, Liang Zhao, Qiming Li, Lei Huang, Yuxuan Gu, Weitao Ma,
Yuan Xu, and Bing Qin. Investigating and mitigating the multimodal hallucination snowballing
in large vision-language models. Association for Computational Linguistics(ACL), 2024.
17

## Page 18

Published as a conference paper at ICLR 2026
A
PROOF OF HALLUCINATION RISK BOUND
A.1
ASSUMPTIONS VALIDATION
We provide theoretical and practical justification for the assumptions adopted in Section 3.2, which
serve to ensure the well-posedness and interpretability of the proposed Hallucination Risk Bound.
These assumptions follow standard practice in NTK-based analyses and stability theory, and are
consistent with the empirical behavior observed in modern large language models.
Setup For completeness, we briefly recall the main notation used in Section 3.2. Let Y denote
the discrete metric space of finite-length token sequences. Let Uh ⊆Rdh be a dh-dimensional
Hilbert space equipped with inner product ⟨·, ·⟩and induced norm ∥· ∥. The task-specific encoder
Φ : Y →Uh is assumed to be LΦ-Lipschitz with respect to dY.
Given input x, the model defines a decoding distribution Pθ(· | x) over Y, and we denote the
embedded random variable by uh := Φ(Y ) where Y ∼Pθ(· | x). For perturbations δ ∈Rr
restricted to the local ball Bρ, the perturbed decoding distribution is denoted Pθ(· | x, δ), and the
mean semantic response map is defined by GY (δ) := EY ∼Pθ(·|x,δ)[Φ(Y )], with Jacobian J =
DGY (0) ∈Rdh×r. The NTK Gram matrix on embedded perturbations is denoted K ∈Rr×r, with
eigenvalues λ1 ≥· · · ≥λr > 0 and condition number κ(K) = λmax/λmin. All expectations
are taken with respect to the specified decoding distribution, and all norms are Euclidean unless
otherwise stated.
Assumption A1 (Integrability and well-defined expectation). Assumption A1 ensures that the
semantic embedding EY ∼pθ(·|x)[Φ(Y )] is well-defined as a Bochner expectation in the finite-
dimensional Hilbert space Uh. The bounded second-moment condition guarantees that the expec-
tation exists and is finite, which is a standard minimal requirement in stochastic analyses of neural
network outputs. Such integrability assumptions are commonly adopted in NTK-based analyses
(Jacot et al., 2018; Lee et al., 2020b), where control of second moments ensures stability of kernel
spectra and well-posedness of linearized approximations.
Assumption A2 (Lipschitz continuity of the encoder Φ). Assumption A2 imposes a controlled
relationship between the discrete sequence space (Y, dY) and the continuous embedding space Uh.
The LΦ-Lipschitz condition ensures that bounded perturbations in edit distance induce proportion-
ally bounded deviations in semantic representation. Such Lipschitz regularity is standard in high-
dimensional learning theory (Vershynin, 2018) and is frequently invoked to establish stability under
structured perturbations in representation learning. Importantly, this assumption is imposed only on
the encoder map Φ, not on the full autoregressive model.
Assumption A3 (Local Fr´echet smoothness of the mean semantic response). Assumption A3
formalizes the local linearization principle underlying NTK theory. By requiring twice Fr´echet dif-
ferentiability of the mean response map GY (δ) = EY ∼Pθ(·|x,δ)[Φ(Y )] within the perturbation ball
Bρ, we ensure that GY admits a controlled second-order expansion with uniform curvature constant
H⋆. This local quadratic remainder bound is consistent with classical finite-width NTK lineariza-
tion results (Lee et al., 2020a; Chizat et al., 2019), while avoiding unrealistic global smoothness
requirements. Crucially, the assumption is imposed only on the expected semantic response, not on
the discrete decoding distribution itself.
Remark.
Collectively, these assumptions provide a bridge between discrete autoregressive gener-
ation and continuous functional analysis. By restricting smoothness and curvature requirements to
the localized perturbation neighborhood Bρ and to the expectation-level map GY , we avoid impos-
ing global regularity conditions over the infinite token space Y. This localization ensures that the
Hallucination Risk Bound is derived under mathematically controlled conditions while remaining
aligned with the practical inference dynamics of large-scale language models.
A.2
PROOF OF SECTION 3.2
We restate the main inequality from Section 3.2. Note that due to the stochastic nature of autore-
gressive decoding, the bound holds with high probability. With probability at least 1 −δ over the
18

## Page 19

Published as a conference paper at ICLR 2026
generation process, the total hallucination risk satisfies:
∥u∗−uh∥≤

1 + kpt log O(P, L) + k ϵmismatch
Signalk

inf
u∈Uh ∥u∗−u∥
|
{z
}
Data-driven term
+ |L| exp

−Kϵ2
C

α
 eβT −1

|
{z
}
Reasoning-driven term
.
(8)
Step 1: Triangle inequality split (Bias-Variance Decomposition).
Let ¯u := E[uh] be the ex-
pected semantic representation under the decoding distribution. By the triangle inequality in Uh, we
decompose the hallucination risk into approximation error (bias) and stochastic residual (variance):
∥u∗−uh∥= ∥u∗−¯u + ¯u −uh∥≤∥u∗−¯u∥+ ∥uh −¯u∥.
Step 2: Approximation term via C´ea’s lemma.
To bound the deterministic approximation error,
we cast the model’s expected representation ¯u as the solution to a variational problem in the Hilbert
space Uh. Let a(u, v) := ⟨u, Kv⟩Uh denote the coercive bilinear form induced by the Neural Tangent
Kernel (NTK) Gram operator K, and let f(v) := ⟨u∗, Kv⟩Uh be the bounded linear functional
defining the target projection. Assuming ¯u acts as the Galerkin projection of the target u∗onto the
trainable hypothesis space, it satisfies the weak formulation a(¯u, v) = f(v) for all v ∈Uh. By C´ea’s
lemma, the projection error is bounded by:
∥u∗−¯u∥≤Λ
γ
inf
u∈Uh ∥u∗−u∥,
where Λ and γ are the continuity and coercivity constants of the NTK-induced bilinear form a(·, ·),
respectively.
Step 3: Variance term via Bernstein concentration.
We now bound the stochastic residual ∥uh−
¯u∥. Let L denote the set of K independent sampled reasoning trajectories used during decoding.
Under our local perturbation assumption (Assumption A3), the deviations of the hidden states are
bounded by the local neighborhood radius ρ. Applying Bernstein’s inequality for bounded random
vectors in a Hilbert space (Vershynin, 2018), the tail probabilities decay exponentially. For an error
tolerance ϵ and an absolute constant C > 0, we have with probability at least 1 −δ:
∥uh −¯u∥≤|L| exp

−Kϵ2
C

α(eβT −1),
where α is a scaling constant, T is the sequence length, and β ≤log σmax bounds the per-step
Jacobian spectral norm.
Step 4: Substitution.
Combining both terms yields the high-probability bound:
∥u∗−uh∥≤Λ
γ
inf
u∈Uh ∥u∗−u∥+ |L| exp

−Kϵ2
C

α(eβT −1).
We now bound the condition ratio Λ/γ via NTK decomposition.
Step 5: Decomposition of NTK Continuity Constant
We decompose the bilinear form a(·, ·)
into three components:
a = a0 + δpt + δmm,
where a0 is the infinite-width baseline kernel, δpt is the perturbation due to pre-training noise, and
δmm is the domain mismatch from fine-tuning. Consequently, the continuity constant satisfies Λ =
Λ0 + ∆pt + ∆mm.
Bounding ∆pt: Following standard matrix concentration bounds for finite-width NTKs (Jacot et al.,
2018), the pre-training deviation scales logarithmically with the network parameters. Let P be the
number of parameters, L the prompt length, and kpt a pre-training scaling constant; we have:
∆pt ≤γkpt log O(P, L).
19

## Page 20

Published as a conference paper at ICLR 2026
Bounding ∆mm: Using spectral generalization bounds under data distribution shift (Lee et al.,
2020b), the mismatch penalty is governed by the task-specific signal strength Signalk, the empirical
mismatch error ϵmismatch, and a scaling constant k:
∆mm ≤γk ϵmismatch
Signalk
.
Substituting both inequalities into the ratio for Λ/γ, and normalizing Λ0/γ ≈1, we obtain:
Λ
γ ≤1 + kpt log O(P, L) + k ϵmismatch
Signalk
.
This completes the proof.
B
HALLUGUARD DERIVATION AND INTERPRETATION
B.1
PRELIMINARIES AND NOTATION
Let K ∈Rr×r be the NTK Gram matrix formed on r light semantic perturbations (see Assumptions
A1-A3 in the main theory section). Denote its eigen decomposition by K = V ΛV ⊤with
Λ = diag(λ1, . . . , λr),
λ1 ≥· · · ≥λr > 0.
Let λmax := λ1, λmin := λr, κ(K) := λmax/λmin, and det(K) = Qr
i=1 λi. Let Φ denote the NTK
feature matrix whose columns span the hypothesis subspace Uh, so that K = Φ⊤Φ, ∥Φ∥2 = √λmax,
and σmin(Φ) = √λmin. For the autoregressive decoder, let Jt be the step-t input–output Jacobian,
and write σmax := supt ∥Jt∥2.
We will use the following two standard inequalities repeatedly:
Maclaurin/AM −−GMoneigenvalues :

r
Y
i=1
λi
1/r
≤1
r
r
X
i=1
λi = tr(K)
r
,
(9)
Submultiplicativity :
∥AB∥2 ≤∥A∥2 ∥B∥2.
(10)
B.2
REPRESENTATIONAL ADEQUACY VIA det(K) WITH EXPLICIT CONSTANTS
Assumptions for this subsection.
Beyond A1–A3, we assume a mild source condition and a
spectral envelope:
S1 (Source condition) Let T denote the infinite-dimensional NTK integral operator.
We
assume there exists a regularity exponent s > 0 and a constant Rs > 0 such that
u∗∈Range(T s). Equivalently, the spectral coefficients satisfy: Pr
i=1
⟨u∗,vi⟩2
λ2s
i
≤R2
s.
This is standard in kernel approximation and encodes RKHS regularity.
S2 (Spectral envelope) Let λ and λ denote uniform upper and lower bounds on the kernel
spectrum. We assume there exist constants 0 < λ ≤λ < ∞and a decay rate α > 1 such
that λi ≤λ for all i, and the tail eigenvalue satisfies λr ≥λ r−α. (Polynomial decay is a
common stylization; other envelopes can be treated similarly.)
Lemma B.1 (Best-approximation error under source condition). Let Uh = span{v1, . . . , vr}. Un-
der S1,
inf
u∈Uh ∥u∗−u∥= ∥u∗−ΠUhu∗∥≤Rs λ s
r+1,
where λr+1 denotes the next-eigenvalue of the infinite-dimensional kernel operator (or, equivalently,
the empirical tail eigenvalue if more perturbations are added).
Proof. Write u∗= P
i≥1 civi with ci = ⟨u∗, vi⟩. By the source condition, ∥u∗−ΠUhu∗∥2 =
P
i>r c2
i ≤P
i>r λ2s
i ·
c2
i
λ2s
i
≤λ2s
r+1
P
i>r
c2
i
λ2s
i
≤λ2s
r+1R2
s.
20

## Page 21

Published as a conference paper at ICLR 2026
To connect the representation error to the empirical NTK Gram matrix K, we leverage the algebraic
relationship between the smallest eigenvalue λr and the determinant.
Lemma B.2 (Lower-bounding λr by det(K)). Suppose λi ≤λ for all i and λr > 0. Then
λr ≥det(K)
λ
r−1
and
λ s
r ≥det(K) s
λ
s(r−1) .
Proof. Since det(K) = Qr
i=1 λi ≤λ
r−1λr, we obtain λr ≥det(K)/λ
r−1. Raising to power s
yields the second inequality.
Theorem B.3 (Determinant-based adequacy bound with explicit constants). Under A1-A3 and S1-
S2, Under A1–A3 and S1–S2, the approximation error is bounded by:
inf
u∈Uh ∥u∗−u∥≤Cd det(K) cd,
with explicit constants independent of the target sequence:
cd = s
r
and
Cd = Rs.
Moreover, if the empirical spectrum satisfies λr ≥λ r−α, one may choose
cd = min



s
r −1 , s
α ·
1
log
 λ
r
det(K)



,
which improves with slower decay (smaller α).
Proof. By Lemma B.1 and the fact that eigenvalues are monotonically decreasing (λr+1 ≤λr), we
have:
inf
u∈Uh ∥u∗−u∥≤Rs λ s
r .
Recall that the determinant of the empirical Gram matrix is the product of its eigenvalues,
det(K) = Qr
i=1 λi. Since λr is the minimum eigenvalue of the rank-r matrix, it follows strictly
that λr
r ≤det(K), which implies λr ≤det(K)1/r. Raising both sides to the power of s yields
λs
r ≤det(K)s/r. Substituting this upper bound into the approximation error gives:
inf
u∈Uh ∥u∗−u∥≤Rs det(K) s/r.
Setting Cd := Rs and cd := s/r completes the proof.
In practice, tracking the direct determinant can cause numerical underflow in high-dimensional
spaces. We use log det(K) via the Cholesky decomposition as our empirical score, aggregating
with z-normalization across components to avoid scale domination by any single dimension.
B.3
ROLLOUT AMPLIFICATION VIA JACOBIAN PRODUCTS (EXACT CONSTANTS)
Theorem B.4 (Amplification bound with exact constant). Let Jt be the step-t Jacobian and σmax :=
supt ∥Jt∥2. Then

T
Y
t=1
Jt

2 ≤
T
Y
t=1
∥Jt∥2 ≤σ T
max.
Defining β := log σmax gives eβT = σT
max, hence
eβT ≤σT
max,
with equality if and only if ∥Jt∥2 = σmax for all t and the top singular directions align across
factors.
Proof. The first inequality is equation 10 applied iteratively. The second is by definition of σmax.
Setting β = log σmax yields equality in the worst case. Alignment of top singular vectors is the
tightness condition for submultiplicativity.
21

## Page 22

Published as a conference paper at ICLR 2026
Token-dependent refinement.
If one defines σt := ∥Jt∥2 and βavg :=
1
T
PT
t=1 log σt, then
 QT
t=1 Jt

2 ≤exp
  P
t log σt

= eβavgT , which is tighter but requires per-step measurements.
B.4
CONDITIONING-INDUCED VARIANCE WITH κ(K)2 SCALING
We now give an explicit projector-perturbation derivation showing the quadratic dependence on the
condition number.
Setup.
Let P := Φ(Φ⊤Φ)†Φ⊤be the orthogonal projector onto Uh; then the linearized output is
uh = Pu∗. Consider a feature perturbation ∆Φ induced by a prefix perturbation δ satisfying
∥∆Φ∥2 ≤LΦ ∥δ∥
(A2/A3).
Let the perturbed projector be eP := (Φ + ∆Φ)
 (Φ + ∆Φ)⊤(Φ + ∆Φ)
†(Φ + ∆Φ)⊤and define
∆P := eP −P.
Lemma B.5 (Projector perturbation bound). There exists an absolute constant CΠ > 0 such that
∥∆P∥2 ≤CΠ
∥Φ∥2
σmin(Φ)2 ∥∆Φ∥2 = CΠ
√λmax
λmin
∥∆Φ∥2 = CΠ κ(K) ∥∆Φ∥2
√λmin
.
Proof idea. Use standard bounds for the perturbation of orthogonal projectors onto column spaces
(e.g., Wedin’s sinΘ theorem and Stewart–Sun, Matrix Perturbation Theory, Thm 3.6). One shows
∥∆P∥2 ≤2 ∥(Φ⊤Φ)†∥2 ∥Φ⊤∆Φ∥2 + O(∥∆Φ∥2
2).
Since ∥(Φ⊤Φ)†∥2 = 1/λmin and ∥Φ⊤∆Φ∥2 ≤∥Φ∥2 ∥∆Φ∥2 = √λmax∥∆Φ∥2, the result follows
for sufficiently small ∥∆Φ∥2, absorbing lower-order terms into CΠ.
Theorem B.6 (Variance amplification with explicit constant). Let uh(Φ) = Pu∗and uh(Φ+∆Φ) =
ePu∗. Then
∥uh(Φ + ∆Φ) −uh(Φ)∥≤CΠ κ(K) ∥∆Φ∥2
√λmin
∥u∗∥.
If ∆Φ is induced by a random prefix perturbation δ with ∥∆Φ∥2 ≤LΦ∥δ∥and E∥δ∥2 = σ2
δ, then
Var[uh] ≤E∥uh(Φ + ∆Φ) −uh(Φ)∥2 ≤cv κ(K)2 ∥δ∥2,
with
cv = C2
Π
L2
Φ ∥u∗∥2
λmin
.
Proof. By Lemma B.5,
∥uh(Φ + ∆Φ) −uh(Φ)∥
=
∥∆P u∗∥
≤
∥∆P∥2∥u∗∥
≤
CΠ κ(K) ∥∆Φ∥2
√λmin ∥u∗∥. Square both sides and take expectation over δ, using ∥∆Φ∥2 ≤LΦ∥δ∥,
to obtain the stated variance bound with the explicit constant cv.
Interpretation.
The κ(K)2 factor arises from two sources: (i) κ(K) from the projector sensitivity
(Lemma B.5), and (ii) 1/λmin from converting ∥∆P∥2 to a mean-squared bound after squaring and
averaging, yielding an overall κ2-scaling in the variance constant.
B.5
CONSOLIDATION: COMPACT SURROGATE CONSISTENT WITH THE RISK
DECOMPOSITION
Combining Theorem B.3, Theorem B.4, and Theorem B.6, we obtain a computable surrogate aligned
with the Hallucination Risk Bound:
Adequacy: det(K)
Amplification: log σmax
Conditioning penalty: −log κ(K)2.
This motivates the score
HALLUGUARD(uh) = det(K) + log σmax −log κ(K)2
with the following explicit, implementation-ready notes:
22

## Page 23

Published as a conference paper at ICLR 2026
• Use log det(K) via Cholesky for stability; replace det in the score with log det if desired
(monotone equivalent).
• Estimate σmax either as supt ∥Jt∥2 or its tighter average form βavg =
1
T
P
t log ∥Jt∥2
(then use βavg in place of log σmax).
• z-normalize each component across a validation set before summation to avoid scale dom-
inance; optionally fit task-specific weights if permitted.
C
EXPERIMENT
C.1
SETUP
Implementation
Framework.
All
experiments
use
PyTorch
and
HuggingFace
Transformers with a fixed random seed for reproducibility.
Unless otherwise noted,
computations run in mixed precision (fp16). Hardware details (A100/H200) are reported once in
the main setup section.
Generation Configuration.
For default evaluation of detectors, we use nucleus sampling with
temperature = 0.5, top-p = 0.95, and top-k = 10, decoding K=10 candidate responses
per input (unless otherwise specified). These decoding trajectories also operationalize semantic per-
turbations as natural variations within the model’s local predictive distribution, thereby instantiating
a semantically proximate neighborhood around the primary response and capturing the local geom-
etry of the reasoning manifold required for NTK construction. For score-guided test-time inference
(Section 4.3), we use beam search (beam size = 10) and score candidate trajectories at each step
with the chosen detector. For stability analysis, HALLUGUARD extracts sentence representations
from the final token at the middle transformer layer (L/2), which empirically preserves semantics
relevant to truthfulness.
NTK-Based Score Computation.
For each set of generations, we form a task-specific NTK fea-
ture matrix and compute the semantic stability score from its eigenspectrum. We add a small ridge
α = 10−3 for numerical stability and compute singular values via SVD.
Perturbation Regularization.
To prevent pathological activations that amplify instability, HAL-
LUGUARD clips hidden features using an adaptive scheme. We maintain a memory bank of N=3000
token embeddings and set thresholds at the top and bottom 0.2% percentiles of neuron activations;
out-of-range values are truncated to attenuate overconfident hallucinations.
Optimization.
Backbone language models are not fine-tuned. We train only HALLUGUARD’s
lightweight projection layers using AdamW with learning rate selected from {1 × 10−5, 5 ×
10−5, 1 × 10−4} and weight decay from {0.0, 0.01}. The best setting is chosen on a held-out
validation split.
Implementation Details.
For score-guided inference we apply beam search with beam size 10,
rescoring candidates stepwise with different hallucination detectors.
Ablation Setup.
All ablations reuse the main paper’s splits, prompts, and decoding; we vary only
HALLUGUARD internals and explicitly control the hallucination base rate. On the generation side,
we modulate prevalence by adjusting temperature/top-p and beam size; to stress the two families,
we increase the prefix perturbation budget ρ and rollout horizon T to amplify reasoning drift, and
(when applicable) toggle retrieval masking to induce data-driven errors. On the detection side, AU-
ROC/AUPRC are threshold-free; when a fixed operating point is needed, we set a decision threshold
τ on the validation set by (i) matching a target predicted-positive rate πtarget via score quantiles or
(ii) fixing a desired FPR (e.g., 1%, 5%, 10%); a cost-sensitive Bayes rule τ =
cFN
cFP + cFN
· 1 −π
π
is
optional when misclassification costs are specified. Unless noted, we toggle one factor at a time and
sweep ρ ∈{0.75, 1.0, 1.5}, T ∈{12, 16, 24}, and the number of semantic probes m ∈{2, 4, 8};
no additional training is performed beyond optional temperature/z-score calibration on the training
split. We report mean±std over 5 seeds.
23

## Page 24

Published as a conference paper at ICLR 2026
C.2
ABLATION STUDY ON −log κ2
To empirically validate the necessity of the stability term −log κ2, we performed a controlled ab-
lation on MATH-500. We systematized the reasoning drift (d) by progressively increasing the per-
turbation budget ρ and rollout horizon T. As shown in Figure 3, the absence of this term leads to
severe instability. While the ablated model (orange dashed line) performs competitively in low-drift
regimes (d < 0.15), it exhibits significant performance volatility as the reasoning task becomes more
complex. In contrast, the full HALLUGUARD score (green solid line) effectively penalizes these ill-
conditioned regimes, maintaining a smooth and robust detection profile. This confirms that −log κ2
functions as an essential spectral regularizer, preventing the score from becoming unreliable under
high-entropy inference states.
Figure 3: Ablation study of the stability term (−log κ2) on MATH500.
Table 6: Ablation on stability term −log κ2 (MATH500).
Method
Pearson R
MSE
HalluGuard
0.985
0.0192
w/o −log κ2
0.8904
0.0381
The error in table 6 nearly doubles without the stability term, confirming that spectral conditioning
is essential for stable reasoning-risk quantification.
C.3
ABLATION STUDY ON SEMANTIC ENCODER Φ
To examine sensitivity to the semantic encoder Φ, we replace the default representation with widely
adopted alternatives, including BERT, SimCSE, and E5. We evaluate across multiple backbone
models and benchmarks.
Table 7 reports AUROC and AUPRC on RAGTruth, GSM8K, and TruthfulQA. Across all settings,
HALLUGUARD consistently outperforms encoder-substituted variants. For example, on QwQ-32B
(RAGTruth), replacing the default encoder with BERT reduces AUROC from 84.59 to 81.44.
These results indicate that the performance gain does not stem from surface semantic similarity
of final outputs. Instead, the method captures geometric structure of reasoning trajectories, which
external encoders cannot fully preserve.
24

## Page 25

Published as a conference paper at ICLR 2026
Table 7: Encoder ablation across backbones and benchmarks (AUROC / AUPRC).
Backbone
Method
RAGTruth
GSM8K
TruthfulQA
AUROC
AUPRC
AUROC
AUPRC
AUROC
AUPRC
GPT-2
HalluGuard
75.51
73.40
72.04
69.88
72.10
68.76
+BERT
72.48
70.12
67.31
64.90
68.02
65.01
+SimCSE
73.21
71.05
68.44
66.02
69.14
66.27
+E5
74.02
71.66
69.12
66.80
70.03
67.10
OPT-6.7B
HalluGuard
80.13
76.77
72.57
70.31
69.59
68.36
+BERT
77.44
74.20
67.95
65.48
66.12
64.80
+SimCSE
78.11
74.83
69.01
66.40
67.08
65.72
+E5
78.66
75.31
70.04
67.25
67.80
66.41
Mistral-7B
HalluGuard
82.31
80.79
80.62
77.30
77.05
73.79
+BERT
79.02
76.91
75.51
72.08
73.14
69.52
+SimCSE
79.88
77.66
76.40
73.01
74.08
70.40
+E5
80.41
78.20
77.12
73.74
74.66
71.05
QwQ-32B
HalluGuard
84.59
81.15
75.81
74.68
74.26
72.76
+BERT
81.44
78.03
70.92
68.90
70.35
68.01
+SimCSE
82.10
78.66
72.10
69.82
71.20
68.70
+E5
82.66
79.12
73.05
70.44
72.02
69.31
LLaMA2-13B
HalluGuard
77.51
75.30
79.01
76.73
78.50
77.56
+BERT
74.26
72.04
73.12
70.60
74.41
72.88
+SimCSE
75.11
72.83
74.20
71.51
75.36
73.54
+E5
75.78
73.44
75.14
72.32
76.10
74.22
C.4
COMPUTATIONAL EFFICIENCY ANALYSIS
To assess practical deployment feasibility, we measured inference latency on an NVIDIA
A100/H200 GPU. Our setup utilizes batched parallel sampling to generate K = 10 trajectories, en-
suring sub-linear scaling of the computational cost. The core HALLUGUARD operations-specifically
feature clipping and computing the NTK score via the Gram matrix-add minimal latency, requiring
less than 1 ms of post-processing time per query.
Figure 4: Per-Question Inference Time (Seconds) on BBH Across Hallucination Detection Methods.
C.5
DETECTION PERFORMANCE ANALYSIS
Across all five model families and three benchmark regimes, HALLUGUARD consistently achieves
state-of-the-art detection performance, particularly in the safety-critical low-FPR regions as shown
in Table 8.
25

## Page 26

Published as a conference paper at ICLR 2026
Figure 5: Per-Question Inference Time (Seconds) on HaluEval Across Hallucination Detection
Methods.
Figure 6: Per-Question Inference Time (Seconds) on Math500 Across Hallucination Detection
Methods.
Figure 7: Per-Question Inference Time (Seconds) on RAGTruth Across Hallucination Detection
Methods.
We additionally expanded our evaluation to include SAPLMA, LLM-Check, and ITI. As shown
in Table 9, HALLUGUARD delivers the strongest performance not only on AUROC/AUPRC but
also on deployment-critical, low-FPR operating points, including F1 and TPR at 5% and 10% FPR.
Across all three benchmarks (RAGTruth, GSM8K, HaluEval) and all backbones (GPT-2 through
QwQ-32B and LLaMA2-13B), HALLUGUARD consistently achieves the highest F1 and the highest
or near-highest TPR under fixed low-FPR constraints. In contrast, SAPLMA and LLM-Check ex-
hibit noticeably lower recall in the stringent 5% FPR regime. These results demonstrate that HAL-
LUGUARD is better aligned with maintaining high detection sensitivity under tight false-positive
budgets, a requirement that is central to reliable hallucination detection in real-world systems.
26

## Page 27

Published as a conference paper at ICLR 2026
Figure 8: Per-Question Inference Time (Seconds) on SQuaD Across Hallucination Detection Meth-
ods.
Figure 9: Per-Question Inference Time (Seconds) on TruthfulQA Across Hallucination Detection
Methods.
C.6
TIGHTNESS OF BOUND
Evaluation of bound tightness.
To rigorously stress-test the Hallucination Risk Bound of The-
orem 3.2, we conducted a controlled synthetic study grounded in the empirical reasoning-depth
distribution of the Snowballing dataset (Zhang et al., 2023). We instantiated empirical hallucination
trajectories by injecting low-variance Gaussian noise into the base components D(T) and R(T),
comparing them against the closed-form theoretical prediction. As illustrated in Figure 10, while
the theoretical curve acts as a conservative upper envelope, it exhibits a nearly parallel growth trajec-
tory to the empirical risk. Crucially, it faithfully captures the exponential curvature and compound-
ing dynamics of the Snowballing Effect. This confirms that the bound possesses high structural
fidelity: it correctly models the scaling law of error propagation across depth ranges, validating its
effectiveness as a ranking proxy despite the absolute numerical offset.
Evaluation of NTK proxy tightness.
To quantitatively validate that our NTK-based proxy faith-
fully captures the amplification behavior of stepwise Jacobians, we conduct a diagnostic experiment
on GPT-2-small (117M), where per-step Jacobian norms are fully tractable. For a held-out set of
GSM8K prompts and decoding steps t ≤18, we compute:
• the empirical stepwise Jacobian magnitude ∥Jt∥2, obtained via automatic differentiation
on the next-token logits, and
• our reasoning-driven NTK proxy, log σmax −log κ2, as defined in Eq. (7), which upper-
bounds the per-step amplification rate and penalizes spectral ill-conditioning of the NTK
Gram matrix.
27

## Page 28

Published as a conference paper at ICLR 2026
Table 8:
Performance comparison on representative benchmarks:
data-centric (RAGTruth),
reasoning-oriented (BBH), and instruction-following (TruthfulQA).
GPT2
OPT-6.7B
Mistral-7B
QwQ-32B
LLaMA2-13B
F1
TPR@10%
TPR@5%
F1
TPR@10%
TPR@5%
F1
TPR@10%
TPR@5%
F1
TPR@10%
TPR@5%
F1
TPR@10%
TPR@5%
RAGTruth
HALLUGUARD
71.22 64.86 51.41
77.03 73.52 59.12
75.19 69.44 59.21
81.91 74.13 63.52
74.66 68.91 57.42
Inside
66.12 59.72 48.31
72.91 70.25 60.37
70.45 68.12 52.41
79.03 74.66 61.09
73.08 70.11 55.26
MIND
58.33 54.11 38.72
62.55 57.81 47.65
71.91 66.74 54.39
64.02 59.12 45.63
68.55 63.50 48.78
Perplexity
55.42 51.20 40.51
63.72 60.13 49.14
69.74 66.51 52.18
70.42 65.41 55.32
60.18 57.01 44.75
LN-Entropy
62.17 57.52 46.44
58.33 52.99 43.28
65.30 61.27 49.92
67.15 62.42 51.33
63.28 59.07 46.14
Energy
59.71 56.23 44.81
60.44 57.18 45.03
63.54 59.42 48.62
72.09 68.15 58.42
66.10 61.33 49.41
Semantic Ent.
57.28 53.42 41.92
69.61 64.81 52.01
67.10 62.44 50.66
66.12 62.15 49.31
64.55 60.18 47.75
Lexical Sim.
61.41 57.09 45.03
65.81 61.44 49.51
62.50 59.12 50.92
70.91 67.53 55.21
66.29 59.88 51.03
SelfCheckGPT
56.22 52.84 40.63
60.79 55.68 45.72
63.12 59.47 48.33
66.54 62.92 51.41
68.21 65.12 53.60
RACE
60.12 56.50 44.90
64.12 59.77 49.22
65.44 61.55 52.73
69.61 66.31 53.92
62.55 59.42 45.66
P(true)
58.91 55.47 42.13
67.44 63.20 51.43
71.22 66.91 54.10
63.44 60.33 49.27
70.18 65.77 52.78
FActScore
62.10 58.21 46.33
59.22 54.14 44.32
63.87 60.77 47.98
68.33 64.02 53.41
65.92 61.37 49.84
BBH
HALLUGUARD
68.33 64.11 56.42
74.91 69.14 62.10
73.22 69.88 57.21
78.55 69.91 61.45
71.10 68.25 59.92
Inside
65.41 61.22 52.83
71.02 67.10 60.21
68.17 64.75 53.92
79.17 72.33 64.22
67.10 63.52 55.91
MIND
54.12 50.22 40.11
57.21 53.44 41.52
63.92 59.88 47.01
61.55 57.14 48.83
65.11 60.22 49.52
Perplexity
52.91 49.33 40.44
61.88 58.12 49.22
62.91 59.42 50.11
59.91 55.72 49.03
60.88 57.41 48.62
LN-Entropy
59.12 55.44 44.92
54.61 51.75 43.18
66.44 63.21 54.09
62.75 59.12 47.52
68.20 64.88 55.41
Energy
53.94 51.22 45.03
56.12 52.14 44.61
64.55 60.11 49.99
68.21 65.12 52.84
66.41 62.77 50.22
Semantic Ent.
57.41 54.32 47.21
61.22 58.42 49.74
63.21 59.10 48.62
63.55 60.24 48.88
64.91 61.44 50.72
Lexical Sim.
50.41 46.77 38.92
60.71 57.11 45.55
59.42 56.88 48.91
70.33 67.10 55.32
58.33 55.42 47.41
SelfCheckGPT
55.21 52.14 43.92
58.10 55.78 46.22
62.82 59.90 50.44
65.22 62.44 54.21
63.44 60.77 52.33
RACE
56.14 53.72 43.88
63.11 59.71 52.81
65.77 62.55 50.72
58.88 55.14 46.18
66.10 62.41 49.81
P(true)
54.31 52.22 44.10
58.22 56.10 48.52
56.91 53.55 43.92
61.40 58.21 46.77
57.33 54.88 45.91
FActScore
56.20 52.42 41.77
55.44 52.12 41.14
61.62 58.22 51.33
59.33 56.42 49.14
63.44 60.22 52.44
TruthfulQA
HALLUGUARD
75.11 71.20 63.21
67.44 64.55 58.12
78.92 74.22 65.33
76.44 72.01 59.92
75.33 69.11 63.08
Inside
71.10 68.55 60.77
61.77 59.44 50.10
63.88 61.33 53.41
69.22 65.10 55.14
62.14 59.94 52.80
MIND
57.44 54.91 45.33
59.92 56.88 48.33
58.72 56.14 47.21
61.21 58.88 52.02
60.44 58.20 49.03
Perplexity
49.52 46.71 38.84
54.12 51.74 43.90
59.72 57.55 46.88
54.44 51.72 42.55
60.33 57.21 47.41
LN-Entropy
57.11 54.88 42.98
55.33 52.41 45.91
59.66 56.22 43.10
60.44 58.02 46.22
61.41 57.17 43.88
Energy
54.11 52.17 38.91
53.44 51.14 36.88
58.21 54.77 49.92
63.02 60.44 51.33
58.41 55.33 50.42
Semantic Ent.
60.08 56.44 44.15
50.14 47.33 35.92
53.74 52.11 37.02
65.33 63.20 50.77
55.02 53.11 38.44
Lexical Sim.
51.22 49.20 39.03
58.72 54.71 48.77
65.71 63.50 53.10
54.77 51.44 45.88
66.41 64.14 54.88
SelfCheckGPT
55.72 53.44 42.78
58.33 55.72 47.14
60.88 57.44 43.91
55.42 54.44 40.77
61.72 59.51 44.10
RACE
52.22 49.88 41.44
63.14 66.88 54.05
70.55 67.11 59.77
55.44 52.11 45.33
71.33 68.22 60.02
P(true)
55.54 52.11 38.82
55.72 52.33 39.22
57.41 53.10 41.22
56.88 54.77 45.55
57.12 53.33 41.88
FActScore
52.91 50.14 40.44
54.11 50.22 41.33
52.88 49.91 42.55
61.55 59.22 44.72
53.41 50.71 43.10
Table 9: Comparison with SAPLMA, LLM-Check and ITI across benchmarks and backbones.
Benchmark
Method
GPT2
OPT-6.7B
Mistral-7B
QwQ-32B
LLaMA2-13B
AUROC
AUPRC
F1
TPR@10%
TPR@5%
AUROC
AUPRC
F1
TPR@10%
TPR@5%
AUROC
AUPRC
F1
TPR@10%
TPR@5%
AUROC
AUPRC
F1
TPR@10%
TPR@5%
AUROC
AUPRC
F1
TPR@10%
TPR@5%
RAGTruth
HALLUGUARD
75.51
73.40
81.22
74.86
61.41
80.13
76.77
77.03
73.52
59.12
82.31
80.79
83.19
79.44
69.21
84.59
81.15
85.91
80.13
63.52
77.51
75.30
74.66
68.91
57.42
SAPLMA
72.80
70.10
72.20
63.50
55.10
78.90
74.20
74.10
68.00
58.20
79.40
77.30
79.00
72.10
60.50
81.00
78.20
79.44
72.80
61.30
74.20
72.10
70.50
61.80
55.90
LLM-Check
68.10
64.50
63.90
55.20
44.80
72.30
68.40
66.50
57.90
46.30
75.20
71.60
67.40
60.30
48.70
76.10
73.20
68.90
61.10
49.50
71.60
68.90
63.20
55.40
46.10
ITI
69.30
65.80
66.10
57.90
47.90
73.10
69.20
68.20
59.80
49.10
76.00
72.50
69.40
61.80
50.90
77.20
74.10
70.50
62.40
51.70
72.80
70.10
65.40
57.10
47.80
GSM8K
HALLUGUARD
72.04
69.88
78.33
74.11
65.42
72.57
70.31
74.91
69.14
62.10
80.62
77.30
80.22
76.88
68.21
75.81
74.68
82.55
78.91
70.45
79.01
76.73
79.10
74.25
67.92
SAPLMA
69.20
66.10
70.10
62.00
54.40
70.80
67.20
71.80
64.10
56.30
77.10
74.00
76.20
69.50
59.80
73.90
71.20
76.50
70.10
60.70
75.40
72.30
74.00
67.10
59.10
LLM-Check
65.40
61.50
62.40
54.10
46.20
68.10
64.30
67.50
59.20
49.80
73.40
69.80
64.90
57.90
48.30
71.20
67.90
67.80
60.30
50.40
72.10
68.50
64.20
56.60
48.00
ITI
66.80
63.00
64.50
56.20
48.70
69.00
65.40
69.20
61.50
51.90
74.20
70.60
67.10
60.80
50.10
72.50
69.20
69.40
62.50
52.30
73.00
69.10
66.10
58.40
49.50
HaluEval
HALLUGUARD
70.42
67.71
75.11
71.20
63.21
71.62
67.88
70.44
67.55
58.12
74.91
72.74
78.92
74.22
65.33
73.93
70.87
76.44
72.01
59.92
78.15
74.15
79.33
75.11
66.08
SAPLMA
67.10
63.20
69.20
62.10
54.00
69.50
65.70
68.30
61.60
53.20
72.00
68.40
75.10
69.30
58.90
71.20
68.10
75.40
70.30
58.50
76.10
72.20
76.80
70.60
60.90
LLM-Check
63.50
59.40
61.10
53.00
44.50
66.80
62.90
65.40
57.50
47.50
70.10
66.30
63.80
57.20
47.10
69.30
65.40
66.20
59.50
49.00
71.50
67.60
63.50
55.90
47.40
ITI
64.80
60.70
63.40
55.20
46.80
67.40
63.50
66.90
58.60
49.40
71.00
67.20
66.10
59.10
48.60
70.20
66.30
68.10
61.10
50.60
72.30
68.20
65.20
57.50
48.70
Figure 11 reports the scatter plot comparing the NTK proxy against empirical ∥Jt∥2 across all
prompts and steps.
Validation of Term Decomposition
To validate the architectural premise of our Hallucination
Risk Bound Section 3.2, we visualize the evolution of the decomposed risk components across rea-
soning depth T on the Snowballing dataset (Zhang et al., 2023). As shown in Figure Figure 12,
the total risk is driven by two distinct dynamic behaviors. The data-driven term (green dotted line)
exhibits linear or near-constant progression, reflecting static retrieval or knowledge-encoding errors
that persist regardless of depth. In contrast, the reasoning-driven term (purple dotted line) demon-
strates exponential amplification consistent with the Snowballing Effect, remaining negligible at
shallow depths but rapidly dominating the total risk as T increases.Crucially, this reveals a phase
transition in hallucination dynamics: at lower depths (T < 15), errors are primarily data-driven,
whereas at higher depths, reasoning instability becomes the governing factor. This dichotomy em-
pirically justifies our hybrid scoring mechanism, confirming that a unified detector must account
28

## Page 29

Published as a conference paper at ICLR 2026
Figure 10: Empirical hallucination risk versus our theoretical bound
Figure 11: The NTK proxy closely tracks empirical Jacobian amplification on GPT-2-small, showing
near-perfect monotonic alignment and a consistent conservative envelope across decoding depth.
for both the static semantic bias and the dynamic rollout instability to be effective across varying
generation lengths.
C.7
CORRELATION OF REASONING-DRIVEN AND DATA-DRIVEN TERMS WITH DIFFERENT
TYPES OF DATASETS
To empirically verify the independence of the proposed risk components, we analyzed their cor-
relation with detection performance across distinct task families. As illustrated in Figure 14 and
Figure 13, we observe a sharp geometric decoupling: the data-driven term aligns strongly with data-
29

## Page 30

Published as a conference paper at ICLR 2026
Figure 12: Risk decomposition across reasoning depth T on Snowballing dataset.
centric benchmarks (e.g., RAGTruth) while showing negligible correlation with reasoning tasks.
Conversely, the reasoning-driven term dominates on reasoning-oriented datasets (e.g., MATH-500).
This double dissociation reinforces the structural validity and orthogonality of our decomposition,
confirming that each term captures a distinct, non-redundant failure mode.
Figure 13: Correlation Between data-driven and reasoning-driven terms and AUROC on Reasoning-
Centric MATH500.
C.8
CASE STUDY
Case Study 1 - GSM8K (Multi-step Arithmetic): Bias →Drift →Snowballing.
Task: “John
saves $3/day for four weeks and buys a $12 toy. How much money does he have left?”
Ground truth: $72.
30

## Page 31

Published as a conference paper at ICLR 2026
Figure 14: Correlation Between data-driven and reasoning-driven terms and AUROC on Data-
Centric RAGTruth.
Length (T)
Model Behavior
HalluGuard Response
T=1–8 Stable setup
Correct restatement and arithmetic planning
Data-driven term dominant; risk flat
T=9–14 Seed error
“4 weeks” →“40 days”
Slight rise in data-driven signal
T=15–22 Propagation
“3 × 40 = 120”
Reasoning-driven share begins to rise
T=23–40 Amplification
Final answer: $108
Reasoning-driven dominates (snowballing)
Table 10: Evolution of hallucination in GSM8K arithmetic reasoning.
Case Study 2 - Long-Document Summarization: Misalignment →Overreach →Fabrication.
Task: Summarize a 5,000-token policy document
Ground truth: Security audit exception applies only to specific log types.
Length (T)
Model Behavior
HalluGuard Response
T=1–20 Accurate extraction
Correct recovery of retention rules
Low risk; strong alignment
T=21–40 Misbinding
Incorrect merge of distant sections
Data-driven signal increases
T=41–95 Drift
Overgeneralized suspension claim
Reasoning-driven share rises
T=96–170 Fabrication
New false rule introduced
Reasoning-driven dominates
Table 11: Evolution of hallucination in long-document summarization.
C.9
COMPARISON WITH INSIDE AND MIND
Inside and MIND serve as empirical uncertainty diagnostics. Inside analyzes covariance spectra of
static representations, while MIND measures temporal variations in hidden states. Both methods
extract post-hoc signals and produce a single uncertainty score.
In contrast, HALLUGUARD derives a structured risk decomposition from generative dynamics, sep-
arating data-driven and reasoning-driven sources via NTK spectral geometry and instability ampli-
fication. This formulation explicitly models compounded reasoning errors.
31

## Page 32

Published as a conference paper at ICLR 2026
We evaluate all methods on the Snowballing benchmark Zhang et al. (2023), which emphasizes
progressive reasoning instability. As shown in Table 12, HALLUGUARD consistently outperforms
Inside and MIND across all backbone models.
Table 12: Comparison with Inside and MIND on the Snowballing benchmark across different back-
bone models (AUROC / AUPRC).
Method
GPT-2
OPT-6.7B
Mistral-7B
QwQ-32B
LLaMA2-7B
LLaMA2-70B
HalluGuard
88.52/82.14
92.63/87.42
94.87/89.66
97.41/95.08
93.28/88.03
97.96/95.37
Inside
74.11/66.39
78.24/70.51
83.32/75.80
87.55/80.47
81.72/73.11
89.03/82.77
MIND
69.42/58.73
74.56/64.37
78.67/68.52
84.03/73.68
77.91/65.89
86.28/78.41
C.10
ADDITIONAL EVALUATION ON MULTIMODAL AND LONG-CONTEXT REGIMES
To evaluate generalization beyond short-form reasoning tasks, we extend experiments to (i) multi-
modal hallucination detection on POPE Li et al. (2023b), and (ii) long-context generation on Gov-
Report Huang et al. (2021) and NarrativeQA s Koˇ cisk´y et al. (2018).
Across all backbone models, HALLUGUARD consistently achieves the strongest AUROC and
AUPRC, demonstrating robustness under multimodal noise and long-range dependency drift.
Table 13: Comparison of methods across different backbone models on POPE(AUROC/AUPRC).
Method
GPT-2
OPT-6.7B
Mistral-7B
QwQ-32B
LLaMA2-7B
LLaMA2-70B
Perplexity
61.12/53.04
68.27/60.18
72.41/64.09
79.36/73.22
70.15/62.31
83.48/76.19
HalluGuard
74.33/68.27
81.22/75.36
86.47/80.51
91.58/86.42
85.39/78.44
94.63/89.27
Inside
70.08/64.12
77.19/70.33
83.44/75.28
89.27/82.36
81.22/74.41
92.51/87.39
MIND
66.17/58.22
73.31/66.14
79.28/71.39
86.44/79.33
77.18/69.27
89.36/83.48
LN-Entropy
63.09/55.11
71.24/62.18
76.37/67.06
84.33/75.29
74.12/65.18
87.42/80.33
Energy
62.14/54.22
69.17/61.26
75.29/66.31
83.41/74.18
73.21/64.33
86.39/79.41
Semantic Ent.
64.18/56.04
72.29/63.14
77.41/68.22
85.48/76.39
75.17/66.41
88.46/81.27
Lexical Sim.
65.24/57.19
73.33/64.21
78.46/69.37
85.52/77.44
76.31/67.29
88.59/82.31
SelfCheckGPT
58.11/50.28
63.22/55.31
67.38/58.24
74.41/66.33
64.27/56.21
78.46/70.39
RACE
69.14/63.17
76.28/69.41
82.33/74.29
88.47/80.36
80.36/73.22
91.44/85.33
P(true)
67.22/59.26
74.31/66.18
80.41/71.33
87.44/79.28
78.29/69.33
90.38/83.41
FActScore
68.19/61.33
75.39/68.22
81.47/73.38
88.52/81.41
79.34/71.48
91.46/85.37
Multimodal Hallucination (POPE).
Table
14:
Comparison
of
methods
across
different
backbone
models
on
GovRe-
port(AUROC/AUPRC).
Method
GPT-2
OPT-6.7B
Mistral-7B
QwQ-32B
LLaMA2-7B
LLaMA2-70B
Perplexity
58.13/49.22
64.41/55.37
67.29/58.46
75.34/66.18
63.28/54.33
78.57/69.41
HalluGuard
72.38/66.41
79.27/72.39
84.46/78.31
90.58/84.42
82.44/76.33
93.62/88.51
Inside
69.17/62.24
76.33/68.41
81.44/73.36
88.42/80.31
79.36/71.29
91.47/85.39
MIND
65.21/56.18
72.41/63.29
77.38/68.33
86.33/77.41
75.27/66.38
88.46/82.24
LN-Entropy
63.12/54.27
70.33/61.22
75.41/65.34
83.39/74.21
72.18/62.33
86.38/79.33
Energy
61.09/52.14
69.18/60.41
74.37/64.28
82.34/73.19
71.26/61.44
85.41/78.36
Semantic Ent.
64.17/55.11
71.22/62.31
76.39/67.42
84.46/75.38
73.24/64.28
87.49/80.36
Lexical Sim.
65.26/56.17
72.38/63.38
76.44/68.41
85.43/76.37
74.22/65.19
87.53/81.44
SelfCheckGPT
55.14/46.29
60.31/51.22
63.44/54.19
70.26/60.41
59.33/49.24
73.41/63.38
RACE
68.28/60.33
75.41/66.29
80.36/72.41
87.42/79.33
78.32/70.24
90.38/84.41
P(true)
66.34/57.22
73.39/64.31
78.48/69.44
86.38/77.41
76.33/67.28
89.44/83.36
FActScore
67.41/59.36
74.42/66.41
79.39/71.46
87.41/78.47
77.47/69.44
90.41/84.38
Long-Context Generation (GovReport).
32

## Page 33

Published as a conference paper at ICLR 2026
Table
15:
Comparison
of
methods
across
different
backbone
models
on
Narra-
tiveQA(AUROC/AUPRC).
Method
GPT-2
OPT-6.7B
Mistral-7B
QwQ-32B
LLaMA2-7B
LLaMA2-70B
Perplexity
56.14/47.22
62.33/53.18
65.41/55.39
72.26/63.41
61.27/51.33
76.38/67.29
HalluGuard
70.36/64.41
77.22/70.37
83.48/76.29
89.53/83.47
81.33/74.36
92.57/87.41
Inside
67.18/60.27
74.39/66.41
80.46/72.31
87.44/79.36
78.41/69.38
90.43/84.32
MIND
63.27/54.18
70.31/61.29
76.33/67.24
84.39/75.41
74.36/64.47
87.41/80.32
LN-Entropy
61.19/52.11
68.27/59.33
73.42/63.21
82.41/73.29
72.14/61.41
85.36/78.44
Energy
60.08/51.14
67.18/58.34
72.37/62.47
81.33/72.41
70.27/60.33
84.44/77.46
Semantic Ent.
63.22/55.09
69.31/61.46
75.44/66.33
83.47/74.41
73.26/63.44
86.47/79.39
Lexical Sim.
64.17/56.22
70.37/62.34
76.41/67.41
84.33/75.44
74.41/65.27
87.46/80.41
SelfCheckGPT
52.14/43.29
57.33/48.31
61.48/51.36
68.41/58.47
56.39/46.31
71.36/61.44
RACE
66.29/58.31
73.42/65.38
79.33/71.28
86.41/78.44
77.28/68.39
89.43/83.38
P(true)
64.31/56.24
71.39/63.33
77.47/68.36
85.38/77.41
75.29/66.33
88.38/82.44
FActScore
65.44/57.36
72.41/64.41
78.52/70.38
86.44/78.33
76.41/68.44
89.44/83.39
Long-Context Generation (NarrativeQA).
D
USAGE OF LLM
Large language models (LLMs) were employed in a limited and transparent manner during the
preparation of this manuscript. Specifically, LLMs were used to assist with linguistic refinement,
style adjustments, and minor text editing to improve clarity and readability. They were not involved
in formulating the research questions, designing the theoretical framework, conducting experiments,
or interpreting results. All scientific contributions-including conceptual development, methodology,
analyses, and conclusions-are the sole responsibility of the authors.
33
