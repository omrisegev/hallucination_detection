---
source_pdf: papers/EDIS paper.pdf
slug: edis-paper
pages: 16
extracted_on: 2026-07-13
---

# EDIS paper

## Page 1

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
Chenghua Zhu * 1 Siyan Wu * 1 Xiangkang Zeng 2 Zishan Xu 3 Zhaolu Kang 4 Yifu Guo 2 Yuquan Lu 1
Junduan Huang 1 Guojing Zhou 1
Abstract
Entropy-based confidence signals are increasingly
leveraged to improve reasoning in large language
models (LLMs), yet existing approaches treat con-
fidence as a static quantity—typically aggregated
over tokens. We show that the temporal evolution
of confidence during generation carries richer in-
formation than aggregate statistics alone. Analyz-
ing token-level entropy trajectories, we identify
characteristic patterns distinguishing correct from
incorrect reasoning: erroneous solutions exhibit
unstable dynamics, including burst spikes (sus-
tained uncertainty growth) and peak-valley spikes
(sharp rebounds following transient confidence).
These patterns persist across models and train-
ing stages, suggesting they reflect intrinsic prop-
erties of reasoning failure rather than superficial
noise. To formalize this observation, we introduce
the Entropy Dynamics Instability Score (EDIS),
a trajectory-level metric quantifying instability
in entropy evolution. EDIS serves as an effec-
tive diagnostic signal for inference-time selection,
substantially improving reasoning accuracy, and
offers a promising direction for training-time sam-
ple curation. Our findings establish entropy dy-
namics as an underexplored yet informative lens
for understanding and improving LLM reasoning.
1. Introduction
Large language models (LLMs) have achieved remarkable
progress on complex reasoning tasks (Wei et al., 2022;
Wang et al., 2022), yet a fundamental challenge persists:
distinguishing correct reasoning from plausible-sounding er-
rors remains difficult without external verification (Farquhar
et al., 2024; Kapoor et al., 2024). A natural approach is
to leverage the model’s own confidence signals—typically
entropy or token-level probabilities—to identify unreliable
*Equal contribution
1South China Normal University,
Guangzhou, China 2Sun Yat-sen University, Guangzhou, China
3Shanghai Jiao Tong University, Shanghai, China 4Peking Uni-
versity, Beijing, China.
Correspondence to:
Guojing Zhou
<gjzhou86@m.scnu.edu.cn>.
Preprint. March 9, 2026.
outputs (Guo et al., 2017; Chen & Mueller, 2024). However,
existing methods treat confidence as a static quantity, ag-
gregating token-level uncertainty into summary statistics or
examining only the final output. Recent evidence suggests
that entropy calibration degrades during autoregressive gen-
eration (Cao et al., 2025), indicating that this static view
may miss important structure. More fundamentally, it over-
looks a key aspect of autoregressive generation: reasoning
unfolds sequentially, and confidence evolves throughout the
process.
In this work, we demonstrate that how confidence evolves
during generation is more informative than its aggregate
value. Through systematic analysis of token-level entropy
trajectories, we uncover a striking pattern: incorrect reason-
ing is not merely associated with higher uncertainty, but
with instability in how uncertainty evolves. As illustrated
in Figure 1, correct reasoning produces relatively smooth
entropy curves where most tokens exhibit low entropy with
few spikes or oscillations. In contrast, incorrect reasoning
shows frequent high-entropy tokens and characteristic in-
stability patterns. We identify two typical failure modes:
burst spikes, where entropy rises steadily over consecutive
tokens as the model becomes progressively confused, and
peak-valley (rebound) spikes, where entropy drops to a lo-
cal minimum before sharply rebounding—indicating false
confidence followed by renewed uncertainty. These insta-
bility patterns are remarkably consistent: across models,
temperatures, and training stages, incorrect responses ex-
hibit 1.7–3.6× more entropy fluctuations than correct ones
(Cohen’s d ≈1.0), suggesting they reflect fundamental
properties of reasoning failure rather than incidental noise.
To operationalize this observation, we introduce the Entropy
Dynamics Instability Score (EDIS), a simple trajectory-level
metric that captures two complementary forms of instabil-
ity: burst spikes (cumulative entropy growth within a sliding
window) and peak-valley spikes (sharp increases from histor-
ical minima). As shown in Figure 2, EDIS distributions for
correct and incorrect responses concentrate around distinct
central values, enabling clear separation. In contrast, mean
entropy—a common baseline—fails to distinguish response
quality, highlighting the value of trajectory-level analysis.
We validate EDIS through extensive experiments on math-
ematical reasoning. For inference-time selection, EDIS-
1
arXiv:2602.01288v2  [cs.LG]  6 Mar 2026

## Page 2

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
Figure 1. Token entropy trajectories for correct (top) and incorrect (bottom) reasoning. Correct responses maintain stable, low entropy,
while incorrect responses exhibit distinctive instability patterns: peak-valley spikes (entropy drops then rebounds) and burst spikes
(progressive entropy rise).
Figure 2. EDIS (left) vs. mean entropy (right) distributions. EDIS
clearly separates correct from incorrect responses, while mean
entropy distributions largely overlap.
based filtering substantially improves answer quality: across
four benchmarks and three models, average accuracy im-
proves from 29.9% to 54.5%—an 82% relative gain—
without requiring verifiers or additional annotations. Com-
pared with alternative confidence measures, EDIS achieves
60.6% overall accuracy versus 51.7% for self-certainty and
50.9% for sequence entropy. We also present preliminary ev-
idence that EDIS can inform training-time sample curation
in reinforcement learning. These results establish entropy
dynamics as an informative signal for assessing reasoning
quality that complements static confidence measures.
Our contributions are as follows:
• We conduct a systematic empirical analysis of entropy
dynamics in LLM reasoning, revealing that incorrect
solutions exhibit characteristic instability patterns—
burst spikes and peak-valley spikes—that persist across
models and training stages.
• We introduce EDIS, a simple and interpretable
trajectory-level metric that quantifies entropy instabil-
ity by combining burst spike detection (cumulative
growth) and peak-valley spike detection (deviation
from historical minima).
• We validate EDIS effectiveness through extensive ex-
periments: EDIS-based selection achieves 82% relative
improvement in accuracy and consistently outperforms
alternative confidence measures. We also present pre-
liminary evidence that EDIS can inform training-time
sample curation.
2. Related Work
Entropy and confidence signals have been extensively stud-
ied in language models, yet existing methods share a com-
mon limitation: they treat confidence as a static quantity,
collapsing the generation process into summary statistics.
We review prior work across three areas and identify the
gap that motivates our trajectory-level approach.
Uncertainty and Confidence Estimation.
Quantifying
model confidence is a long-standing challenge in machine
learning (Guo et al., 2017; Desai & Durrett, 2020). In
language models, common approaches aggregate token-
level probabilities or entropy into sequence-level scores
for calibration and selective prediction (Kadavath et al.,
2022).
The simplest approach computes mean entropy
¯H =
1
T
PT
t=1 Ht across tokens, capturing average uncer-
2

## Page 3

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
tainty but discarding temporal information. Semantic en-
tropy (Farquhar et al., 2024) groups semantically equivalent
outputs to reduce spurious variability, enabling meaning-
aware hallucination detection, but still produces a single
scalar per generation. Self-certainty (Kang et al., 2025)
measures probability mass concentration in the distribution
tail, providing a token-level confidence signal. Recent work
on entropy minimization (Prabhudesai et al., 2025; Zhao
et al., 2025) demonstrates that lower entropy correlates with
reasoning accuracy, with tokens near final answers exhibit-
ing the strongest signal. However, all these methods collapse
the generation process into summary statistics—averaging
entropy or examining only specific tokens—discarding the
temporal structure of how confidence evolves throughout
reasoning.
Inference-Time Scaling.
Scaling compute at inference
time has emerged as an effective strategy for improving
reasoning (Snell et al., 2024; Brown et al., 2024). Common
approaches generate multiple candidates and select among
them via majority voting (Wang et al., 2022), verifier-based
reranking (Cobbe et al., 2021; Lightman et al., 2023), or
confidence-based filtering. Entropy signals have been ap-
plied to early stopping in chain-of-thought reasoning (Sui
et al., 2025) and selective abstention (Xie et al., 2023). Yet
these methods assess reliability from aggregate confidence
scores, implicitly assuming that a single scalar suffices to
characterize reasoning quality. Process-aware verification
validates intermediate steps (Lightman et al., 2023) but re-
quires external verifiers or human annotations, limiting scal-
ability.
Entropy Signals in Training.
Entropy also plays a critical
role in reinforcement learning for LLM reasoning. Policy
entropy collapse—where models become excessively deter-
ministic during training—limits performance scaling (Cui
et al., 2025), motivating work on entropy-aware objectives
such as maximum entropy RL (Haarnoja et al., 2018). Com-
plementary work defines intrinsic rewards based on en-
tropy or confidence, enabling unsupervised reasoning with-
out external supervision: RENT (Prabhudesai et al., 2025)
trains models to minimize token-level entropy, while Zhao
et al. (2025) rewards self-certainty maximization. These
approaches directly modify the training objective by using
entropy or confidence as the reward. In contrast, EDIS pre-
serves the original reward signal and uses entropy dynamics
to curate training data—strengthening more informative
sequences and down-weighting less informative ones.
3. Preliminaries
We begin by establishing the formal framework for analyz-
ing entropy dynamics in autoregressive language models
and introduce the reinforcement learning algorithm used in
our experiments.
Token-Level Entropy.
Consider an autoregressive lan-
guage model parameterized by θ that generates a response
y = (y1, y2, . . . , yT ) conditioned on a prompt x.
At
each generation step t, the model produces a distribution
πθ(yt | x, y<t) over the vocabulary V. We quantify the
model’s uncertainty at position t via the token entropy:
Ht = −
X
v∈V
πθ(v | x, y<t) log πθ(v | x, y<t).
(1)
Low entropy indicates high confidence, with probability
mass concentrated on a small set of tokens; high entropy
reflects uncertainty, with probability spread more uniformly
across the vocabulary.
Entropy Trajectory.
While prior work typically aggre-
gates token-level entropy into scalar summaries (see Sec-
tion 2), we argue that the temporal structure of entropy
evolution carries important information. We define the en-
tropy trajectory as the ordered sequence:
H = (H1, H2, . . . , HT ),
(2)
where each Ht is computed according to Eq. (1). This repre-
sentation preserves the dynamics of uncertainty throughout
generation, enabling analysis of how confidence evolves
rather than merely how confident the model is on average.
As we demonstrate empirically, this trajectory-level perspec-
tive reveals diagnostic patterns that distinguish correct from
incorrect reasoning.
Group Relative Policy Optimization.
We adopt Group
Relative Policy Optimization (GRPO) (Shao et al., 2024)
as our RL training framework. For each prompt x, GRPO
samples a group of G responses {y(1), . . . , y(G)} from the
old policy πθold and computes a reward r(i) for each. The
objective maximizes a clipped surrogate with group-relative
advantages:
JGRPO(θ) = E
"
1
G
G
X
i=1
min
 ρ(i) ˆA(i), ¯ρ(i) ˆA(i)
#
,
(3)
where ρ(i) = πθ(y(i)|x)/πθold(y(i)|x) is the importance
ratio, ¯ρ(i) = clip(ρ(i), 1 −ϵ, 1 + ϵ) is the clipped ratio, and
the advantage is estimated relative to the group:
ˆA(i) = r(i) −µG
σG
,
(4)
with µG and σG being the mean and standard deviation of
rewards within the group. This eliminates the need for a
separate critic network while providing stable advantage
estimates through within-group normalization.
3

## Page 4

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
Figure 3. EDIS-based sample curation for RL training.
4. Methods - Entropy Dynamics Instability
Score (EDIS)
We first describe the empirical phenomena that motivate our
approach, then present the formal definition of the Entropy
Dynamics Instability Score (EDIS).
4.1. Characteristic Instability Patterns
Analyzing rollouts from Qwen2.5-Math-1.5B across train-
ing checkpoints and temperatures reveals that incorrect so-
lutions exhibit significantly more entropy spikes than cor-
rect ones (1.7–3.6× more; Appendix B). Beyond aggregate
counts, we identify two characteristic temporal patterns that
reliably distinguish incorrect reasoning:
Burst Spikes.
A recurring signature of incorrect reason-
ing is sustained entropy increase over consecutive tokens
(Figure 1, bottom-right). Rather than a single abrupt jump,
entropy rises steadily across a window of w tokens, indi-
cating progressive deterioration of model confidence. We
formalize this pattern by counting positions where cumula-
tive entropy growth exceeds a threshold:
Sburst =
T −w
X
t=1
I
 Ht+w −Ht > τb

,
(5)
where I(·) is the indicator function, w is the window size,
and τb is the threshold for detecting significant entropy
growth. This captures the intuition that “the more the model
generates, the more confused it becomes”—a hallmark of
reasoning gone astray.
Peak-Valley Spikes.
A second pattern involves false con-
fidence followed by renewed uncertainty—a characteristic
V-shaped trajectory (Figure 1, bottom-left). The running
minimum mins<t Hs tracks the most confident state the
model has achieved so far; when current entropy Ht rises
significantly above this baseline, it indicates that previously
attained confidence has eroded. We count such events as:
Srebound =
T
X
t=2
I
 Ht −min
s<t Hs > τr

,
(6)
where τr is the threshold for detecting significant rebound
from historical minima. This formulation inherently cap-
tures V-shaped dynamics: the running minimum only be-
comes low if entropy has previously decreased (the descent
into the valley), and the threshold is only exceeded when
entropy subsequently rises (the ascent out of it). The pattern
signals that the model reached a confident state but then
encountered renewed difficulty.
Both patterns are qualitatively distinct from correct reason-
ing, which exhibits smoother entropy evolution with fewer
abrupt transitions. Appendix A provides token-level visual-
izations illustrating these dynamics.
4.2. Metric Definition
The instability patterns described above suggest that rea-
soning quality can be diagnosed from entropy trajectory
structure. We operationalize this insight with the Entropy
Dynamics Instability Score (EDIS), a trajectory-level metric
that combines spike frequency with overall variance:
EDIS(H) = S(H) ·
 1 + Var(H)

,
(7)
where S(H) = 1
2(Sburst + Srebound) denotes the combined
spike score and
Var(H) = 1
T
T
X
t=1
(Ht −¯H)2,
¯H = 1
T
T
X
t=1
Ht
(8)
is the entropy variance. The multiplicative formulation cap-
tures the intuition that reasoning is most unstable when spike
events co-occur with high overall variance. Lower EDIS
indicates more stable reasoning; EDIS ≈0 corresponds to
smooth, confident generation.
4.3. EDIS for Reinforcement Learning
While the primary application of EDIS is inference-time se-
lection, the same principle—that entropy stability indicates
reasoning quality—may also inform training. We present
a preliminary exploration of using EDIS to curate training
samples in RL, as a proof-of-concept rather than a fully
optimized approach.
4

## Page 5

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
The intuition is straightforward: not all training samples
are equally informative. Confident correct responses (low
EDIS) represent reliable reasoning worth reinforcing, while
struggling incorrect ones (high EDIS) reveal genuine dif-
ficulties worth learning from. In contrast, lucky correct
guesses (high EDIS) and systematic failures (low EDIS)
provide weaker learning signals. We explore two mech-
anisms that leverage this structure (Figure 3): sequence
filtering retains only high-signal trajectories, and sequence
weighting assigns differential importance to all samples.
Sequence Filtering.
The simplest approach retains only
extreme trajectories: the most stable correct responses (low-
est EDIS) and the most unstable incorrect ones (highest
EDIS). In practice, we oversample m · n candidates per
prompt and alternately select from these extremes until n
samples remain, discarding ambiguous cases entirely.
Sequence Weighting.
Rather than discarding samples, we
can assign differential weights based on EDIS. Since raw
EDIS scores often exhibit skewed distributions with long
tails, we apply a log transformation to compress the range
and then standardize to z-scores: zi = (log(EDISi + 1) −
µ)/σ, where µ and σ are batch statistics. This normal-
ization ensures consistent weighting across batches with
different EDIS distributions. After standardization, zi > 0
indicates above-average instability and zi < 0 indicates
below-average instability. A correctness-dependent transfor-
mation ensures informative samples receive higher weights:
si =
(
−zi
if correct
zi
if incorrect
(9)
For correct trajectories, we negate to favor stability (low
EDIS →high weight); for incorrect trajectories, we preserve
the sign to favor instability (high EDIS →high weight).
Weights are computed via wi = softmax(si/α) · n, where
α controls concentration. To preserve gradient balance,
we normalize separately within the correct and incorrect
groups:
wnorm
i
=
wi
P
j∈Gi wj
· |Gi|,
(10)
where Gi denotes the correctness group (correct or incorrect)
containing sample i. The weighted advantage becomes
˜Ai = Ai · wnorm
i
. We apply weighting only to prompts with
mixed outcomes.
5. Experiments
This section evaluates EDIS in two settings: inference-
time selection (Sections 5.1–5.3) and reinforcement learning
(Section 5.4).
5.1. Best-of-N Selection
The first question is whether EDIS captures meaningful sig-
nal for inference-time selection. Unlike verifier-based meth-
ods that require additional training or supervision, EDIS
leverages only the model’s internal uncertainty dynamics. If
EDIS reliably distinguishes response quality, filtering from
larger candidate pools should yield systematic accuracy
gains.
Setting.
Experiments
span
four
mathematical
rea-
soning benchmarks—GSM8K (Cobbe et al., 2021),
MATH (Hendrycks et al., 2021), AMC23 (knoveleng, 2025),
and AIME24 (Hugging Face H4, 2025)—using three mod-
els: Qwen2.5-Math-1.5B (Qwen Team, 2024a; Yang et al.,
2024), Qwen3-4B-Instruct (Qwen Team, 2025; Yang et al.,
2025), and Qwen2.5-Math-7B (Qwen Team, 2024b; Yang
et al., 2024). For GSM8K and MATH, we randomly sample
100 problems; for AMC23 and AIME24, we use the full test
sets. All experiments use three sampling temperatures (0.2,
0.6, 1.0), with results averaged across temperatures. For
each problem, we generate N = m · k candidates (k = 8,
m ∈{1, 2, 4, 8, 16}), rank by EDIS, and retain the k most
stable responses (lowest EDIS).
Results.
Figure 4 shows remarkably consistent improve-
ments: across all three models, four benchmarks, and three
metrics (average accuracy, EDIS-best, and majority voting),
accuracy increases monotonically with the oversampling
multiplier. The gains are substantial, particularly for mod-
els with lower baseline performance. Aggregating across
benchmarks, Qwen2.5-Math-1.5B improves from 29.9% to
54.5% in average accuracy as m increases from 1 to 16—a
gain of 24.6 percentage points that nearly doubles the base-
line. Similarly, Qwen2.5-Math-7B improves from 40.9% to
61.9% (+21.0 pp). Even the stronger Qwen3-4B-Instruct,
starting at 58.8%, achieves consistent gains to 62.2% (+3.4
pp). The pattern holds for majority voting, where Qwen2.5-
Math-1.5B improves from 39.4% to 62.6% (+23.2 pp) and
Qwen2.5-Math-7B from 55.9% to 69.8% (+13.9 pp). No-
tably, even EDIS-best accuracy improves (e.g., 46.3% to
54.1% for Qwen2.5-Math-1.5B), indicating that EDIS fil-
tering enriches the candidate pool with higher-quality re-
sponses rather than merely improving answer aggregation.
These results confirm that EDIS reliably indicates reasoning
quality, enabling substantial improvements without exter-
nal supervision. Detailed per-model breakdowns across all
temperatures are provided in Appendix C.
5.2. Comparison with Other Selection Methods
To evaluate the effectiveness of EDIS relative to other se-
lection methods, we compare against several baselines:
Mean (unweighted average), Majority Voting (most fre-
quent answer), Sequence Entropy (mean token-level en-
5

## Page 6

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
Figure 4. EDIS-based best-k-of-N selection across models and benchmarks (k = 8). Accuracy improves consistently as the oversampling
multiplier m increases. Results averaged across three temperatures (0.2, 0.6, 1.0).
tropy ¯H = 1
T
PT
t=1 Ht, lower indicates higher confidence),
and Self-Certainty (SC) (Kang et al., 2025), which mea-
sures confidence via KL divergence between the predicted
distribution and a uniform distribution over the vocabulary,
aggregated across tokens; higher values indicate the model
concentrates probability mass on fewer tokens, suggesting
greater certainty.
Setting.
For confidence-based methods (Entropy, SC,
EDIS), we use Score-Weighted Borda aggregation: each
response casts a weighted vote for its predicted answer, with
the highest-weighted answer selected. For metrics where
lower values indicate higher confidence (Entropy, EDIS), we
use inverse weighting wi = (si +ϵ)−1 with ϵ = 0.1; for SC,
we use wi = si directly. Experiments use Qwen2.5-Math-
1.5B across four benchmarks, generating m ∈{4, 8, 16}
candidates per problem, with results averaged across three
temperatures.
Results.
Table 1 shows that EDIS consistently achieves
the highest overall accuracy across all candidate pool sizes:
57.0% at m = 4, 59.5% at m = 8, and 60.6% at m =
16—outperforming the next-best method (SC) by 6.3 to 8.9
percentage points. The gains are particularly pronounced on
GSM8K (72.3% vs. 56.0% for SC at m = 16) and MATH
(62.3% vs. 54.0%).
Notably, EDIS’s advantage over Sequence Entropy (60.6%
vs. 50.9%) shows that how entropy evolves matters more
than its average value. EDIS achieves best or tied-best
performance on 9 of 12 dataset-m combinations; Sequence
Entropy occasionally wins on AMC23, where the limited
Table 1. Comparison of selection methods.
m
Dataset
Mean
Maj Vote
Entropy
SC
EDIS
4
GSM8K
36.0
47.3
53.3
56.3
67.3
MATH
30.0
46.3
50.7
53.0
58.0
AMC23
38.9
56.7
58.3
56.7
55.8
AIME24
7.2
16.7
18.9
16.7
21.1
Overall
31.0
44.9
49.3
50.7
57.0
8
GSM8K
36.3
49.7
56.7
58.7
72.3
MATH
29.5
46.0
50.3
52.3
60.0
AMC23
38.2
59.2
60.0
59.2
57.5
AIME24
7.5
16.7
16.7
17.8
17.8
Overall
30.8
46.1
50.4
51.9
59.5
16
GSM8K
35.6
49.0
55.0
56.0
72.3
MATH
29.5
47.3
52.0
54.0
62.3
AMC23
37.6
57.5
60.8
60.0
55.8
AIME24
6.9
17.8
20.0
18.9
22.2
Overall
30.4
46.2
50.9
51.7
60.6
test set size may reduce statistical power. These results
demonstrate that EDIS provides a reliable confidence signal
that scales effectively with increased inference compute.
Full results broken down by temperature are reported in
Appendix D.
5.3. EDIS vs. Mean Entropy
To further study the predictive power of EDIS, this section
compares it against mean entropy at the individual sequence
level. The analysis covers 26,356 valid responses (excluding
no-answer responses) from Qwen2.5-Math-1.5B across four
benchmarks at three temperatures.
6

## Page 7

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
Figure 5. EDIS vs. Mean Entropy as correctness predictors. (a-b)
EDIS provides clearer separation. (c) EDIS achieves higher AUC
(0.804 vs. 0.673). (d) EDIS maintains advantage across retention
rates.
Correlation Analysis.
EDIS and mean entropy are mod-
erately correlated (Pearson r = 0.58, Spearman ρ = 0.66),
indicating they capture related but distinct information. Crit-
ically, their correlations with correctness differ: mean en-
tropy shows higher linear correlation (Pearson r = −0.19
vs. −0.10), but EDIS achieves substantially stronger rank
correlation (Spearman ρ = −0.52 vs. −0.30). This reveals
that EDIS captures non-linear ranking relationships crucial
for selection tasks.
ROC-AUC Performance.
Figure 5(c) shows ROC curves.
EDIS achieves an AUC of 0.804—correctly ranking a ran-
dom correct-incorrect pair 80.4% of the time—compared
to 0.673 for mean entropy, a gap of 13.1 points. The EDIS
curve rises steeply at low false positive rates, indicating
effective separation at aggressive thresholds.
Selection Accuracy.
When selecting responses by lowest
score (highest confidence), EDIS maintains consistent ad-
vantages (Figure 5(d)). At top 10% retention, EDIS achieves
91.1% accuracy versus 61.0% for entropy—a 30-point gap.
The advantage persists at 20% (+21.7 pp), 30% (+14.6 pp),
and 50% (+9.0 pp) retention, demonstrating that trajectory
dynamics capture quality signals that aggregate statistics
miss.
5.4. EDIS for Reinforcement Learning
To examine whether EDIS can provide useful training sig-
nals for RL, we design experiments comparing training with
and without EDIS guidance. We also investigate whether
EDIS-based test-time selection remains beneficial as RL
training proceeds. Our primary goal is to verify that EDIS
meaningfully distinguishes informative trajectories—not to
optimize RL performance.
Setting.
All experiments use Qwen2.5-Math-1.5B trained
on NuminaMath-20K for 500 steps with GRPO (k = 8 re-
sponses per prompt), validating on AMC23 every 10 steps.
We conduct two experiment sets: (1) at T = 0.6, we com-
pare full EDIS-informed training against the baseline; (2)
at T = 0.2, we perform a complete ablation to isolate each
component’s contribution.
Configurations.
Five settings progressively incorporate
EDIS: (1) Standard GRPO: baseline without EDIS; (2) +
Oversampling: adds test-time EDIS selection (4× candi-
dates) without training-time signals; (3) + EDIS filtering:
adds training-time filtering (m = 1.25; Section 4.3), retain-
ing only the most stable correct and most unstable incorrect
responses; (4) + EDIS weighting: adds training-time weight-
ing (α = 1.8), assigning differential importance via soft-
max over signed z-scores—lower α concentrates weight on
extremes, higher α yields more uniform weighting; (5) Fil-
tering + Weighting: combines both mechanisms (m = 1.25,
α = 1.8)—the full EDIS-informed configuration used at
T = 0.6. Configurations (2)–(5) share the same test-time
EDIS selection; differences lie solely in training-time sig-
nals.
Table 2. EDIS as a training signal (best validation accuracy).
T = 0.6: full EDIS-informed vs. baseline. T = 0.2: component
ablation.
T
Configuration
maj@8
mean@8
0.6
Standard GRPO
60.8%
53.8%
EDIS-informed
66.2% (+5.4)
61.9% (+8.1)
0.2
Standard GRPO
59.1%
55.0%
+ Oversampling
62.1% (+3.0)
58.4% (+3.4)
+ EDIS filtering
62.6% (+3.5)
60.6% (+5.6)
+ EDIS weighting
66.8% (+7.7)
62.2% (+7.2)
Filtering + Weighting
66.5% (+7.4)
62.8% (+7.8)
Results.
Figure 6 and Table 2 show consistent patterns
across both temperatures. At T = 0.6, full EDIS-informed
training yields substantial gains over the baseline (+5.4
pp maj@8, +8.1 pp mean@8). The T = 0.2 ablation
isolates each component: test-time selection alone (over-
sampling) provides meaningful benefit (+3.0 pp maj@8),
confirming that EDIS remains useful for selecting among
RL-trained outputs. Training-time filtering adds modest ad-
ditional signal (+3.5 pp total), while weighting achieves the
largest gains (+7.7 pp maj@8). Combining both yields the
best mean@8 (+7.8 pp). Beyond accuracy, EDIS-informed
training produces dramatically lower entropy (0.07–0.09 vs.
0.16–0.18) and shorter responses (453–525 vs. 620–646 to-
7

## Page 8

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
Figure 6. Validating EDIS as a training signal. Top row: At temperature T = 0.6, EDIS-informed training (filtering + weighting) vs.
standard GRPO baseline. Bottom row: Complete ablation at T = 0.2, isolating the contribution of each EDIS component.
kens). EDIS maintains stable discriminative power through-
out training, with the spike ratio between incorrect and cor-
rect responses at 1.9–2.7× across all 500 steps (Appendix E;
additional training dynamics in Appendix F).
Interpretation.
These results connect directly to EDIS’s
core insight: entropy dynamics distinguish trajectories by
how reasoning unfolds, not just final correctness. The lower
entropy and shorter responses reflect more focused reason-
ing, where the model avoids uncertainty cascades charac-
teristic of incorrect trajectories. Notably, entropy reduction
is more pronounced at T = 0.6 (59% vs. 50% at T = 0.2),
suggesting EDIS provides greater benefit when baseline gen-
eration exhibits higher variability. The gap between filtering
(+3.5 pp) and weighting (+7.7 pp) reveals an important
distinction: both use the same number of training samples,
but filtering reshapes the EDIS distribution by retaining only
extremes (low-EDIS correct, high-EDIS incorrect) and dis-
carding ambiguous cases, while weighting preserves the full
distribution with differential importance. This suggests that
“middle ground” samples—moderately stable correct and
moderately unstable incorrect responses—still carry useful
gradient signal when appropriately weighted.
6. Limitations and Future Work
Limitations.
This investigation focuses on mathemati-
cal reasoning, where correctness is objectively verifiable.
Whether the instability patterns transfer to other reasoning-
intensive domains—code generation, scientific reasoning,
logical deduction—remains to be validated. Our RL ex-
periments serve as a proof-of-concept; more sophisticated
integration strategies may yield larger gains.
A more fundamental limitation concerns EDIS’s generality
across models. While the qualitative patterns appear consis-
tently, optimal thresholds and parameters vary across model
families and sizes. Window sizes, rebound thresholds, and
spike weightings require calibration for new models, as en-
tropy dynamics depend on model-specific factors such as
vocabulary size and training distribution.
Future Work.
Two directions seem particularly promis-
ing. First, token-level credit assignment: extending EDIS to
identify which tokens contribute most to instability could en-
able fine-grained feedback for process reward models. Sec-
ond, unsupervised process supervision: extending trajectory-
level signals to step-level analysis—by segmenting at rea-
soning boundaries and computing local instability—could
help bootstrap process reward models without ground-truth
step labels.
7. Conclusion
We introduced EDIS, a trajectory-level metric that captures
instability patterns in entropy evolution during LLM rea-
soning. The central insight is that reasoning quality can be
diagnosed from how confidence evolves during generation,
not just its average value. By shifting from static to dynamic
analysis, EDIS extracts richer signal from token-level data
that prior methods reduce to summary statistics. The char-
acteristic instability patterns—burst spikes and peak-valley
spikes—persist across models, temperatures, and training
stages, suggesting they reflect fundamental properties of
reasoning failure. EDIS achieves an 82% relative accuracy
improvement for inference-time selection and up to +7.7
percentage points gains for RL training, consistently outper-
forming alternative confidence measures.
8

## Page 9

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
References
Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V.,
R´e, C., and Mirhoseini, A. Large language monkeys:
Scaling inference compute with repeated sampling. arXiv
preprint arXiv:2407.21787, 2024.
Cao, S., Valiant, G., and Liang, P. On the entropy calibration
of language models. arXiv preprint arXiv:2511.11966,
2025.
Chen, J. and Mueller, J. Quantifying uncertainty in answers
from any language model and enhancing their trustwor-
thiness. In Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1:
Long Papers), pp. 5186–5200, 2024.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
R., et al. Training verifiers to solve math word problems.
arXiv preprint arXiv:2110.14168, 2021.
Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li,
H., Fan, Y., Chen, H., Chen, W., et al. The entropy mech-
anism of reinforcement learning for reasoning language
models. arXiv preprint arXiv:2505.22617, 2025.
Desai, S. and Durrett, G. Calibration of pre-trained trans-
formers. arXiv preprint arXiv:2003.07892, 2020.
Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. Detecting
hallucinations in large language models using semantic
entropy. Nature, 630(8017):625–630, 2024.
Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On
calibration of modern neural networks. In International
conference on machine learning, pp. 1321–1330. PMLR,
2017.
Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft
actor-critic: Off-policy maximum entropy deep reinforce-
ment learning with a stochastic actor. In International
conference on machine learning, pp. 1861–1870. Pmlr,
2018.
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,
S., Tang, E., Song, D., and Steinhardt, J. Measuring math-
ematical problem solving with the math dataset. arXiv
preprint arXiv:2103.03874, 2021.
Hugging Face H4. Aime 2024: Problems from aime i and
aime ii (2024). Hugging Face Datasets, 2025. URL
https://huggingface.co/datasets/Hugg
ingFaceH4/aime_2024.
Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain,
D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma,
N., Tran-Johnson, E., et al. Language models (mostly)
know what they know. arXiv preprint arXiv:2207.05221,
2022.
Kang, Z., Zhao, X., and Song, D. Scalable best-of-n selec-
tion for large language models via self-certainty. arXiv
preprint arXiv:2502.18581, 2025.
Kapoor, S., Gruver, N., Roberts, M., Collins, K., Pal, A.,
Bhatt, U., Weller, A., Dooley, S., Goldblum, M., and
Wilson, A. G. Large language models must be taught
to know what they don’t know.
Advances in Neural
Information Processing Systems, 37:85932–85972, 2024.
knoveleng. Amc-23: A 40-problem test set from amc 12
(2023). Hugging Face Datasets, 2025. URL https:
//huggingface.co/datasets/knoveleng/
AMC-23.
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker,
B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and
Cobbe, K. Let’s verify step by step. In The Twelfth
International Conference on Learning Representations,
2023.
Prabhudesai, M., Chen, L., Ippoliti, A., Fragkiadaki, K.,
Liu, H., and Pathak, D. Maximizing confidence alone
improves reasoning. arXiv preprint arXiv:2505.22660,
2025.
Qwen Team. Qwen2.5-math-1.5b. Hugging Face Model
Card, 2024a. URL https://huggingface.co/Q
wen/Qwen2.5-Math-1.5B.
Qwen Team. Qwen2.5-math-7b. Hugging Face Model Card,
2024b. URL https://huggingface.co/Qwen/
Qwen2.5-Math-7B.
Qwen Team.
Qwen3-4b-instruct-2507.
Hugging Face
Model Card, 2025. URL https://huggingfac
e.co/Qwen/Qwen3-4B-Instruct-2507.
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang,
H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Push-
ing the limits of mathematical reasoning in open language
models. arXiv preprint arXiv:2402.03300, 2024.
Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm test-
time compute optimally can be more effective than scal-
ing model parameters. arXiv preprint arXiv:2408.03314,
2024.
Sui, Y., Chuang, Y.-N., Wang, G., Zhang, J., Zhang, T.,
Yuan, J., Liu, H., Wen, A., Zhong, S., Zou, N., et al.
Stop overthinking: A survey on efficient reasoning for
large language models. arXiv preprint arXiv:2503.16419,
2025.
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang,
S., Chowdhery, A., and Zhou, D. Self-consistency im-
proves chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171, 2022.
9

## Page 10

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi,
E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting
elicits reasoning in large language models. Advances in
neural information processing systems, 35:24824–24837,
2022.
Xie, Y., Kawaguchi, K., Zhao, Y., Zhao, J. X., Kan, M.-
Y., He, J., and Xie, M. Self-evaluation guided beam
search for reasoning. Advances in Neural Information
Processing Systems, 36:41618–41650, 2023.
Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu,
D., Tu, J., Zhou, J., Lin, J., et al. Qwen2. 5-math techni-
cal report: Toward mathematical expert model via self-
improvement. arXiv preprint arXiv:2409.12122, 2024.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical
report. arXiv preprint arXiv:2505.09388, 2025.
Zhao, X., Kang, Z., Feng, A., Levine, S., and Song, D.
Learning to reason without external rewards.
arXiv
preprint arXiv:2505.19590, 2025.
10

## Page 11

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
A. Token-Level Visualization of Entropy Dynamics
Figure 7 provides a token-level view of the entropy dynamics illustrated in Figure 1. Each token is rendered with background
color encoding either entropy magnitude (left columns) or spike status (right columns).
Entropy Heatmap.
Color ranges from green (low entropy) through yellow to red (high entropy), visualizing the model’s
confidence at each generation step.
Spike Heatmap.
Light green indicates no spike; yellow indicates a single spike type (burst or peak-valley); orange
indicates both spike types co-occur at that position.
Qualitative Analysis.
Case 1 (correct reasoning) exhibits predominantly green tokens with sparse, isolated perturbations.
Case 2 (incorrect reasoning) shows extensive yellow/red regions in the entropy heatmap and pervasive yellow/orange regions
in the spike heatmap. Notably, hallucinated tokens (e.g., “cheduler”, Thai characters) concentrate in high-entropy, high-spike
regions.
Discrimination Ratio.
EDIS achieves a 14.0× discrimination ratio between the two cases (110.8 vs. 7.9), compared to
only 3.6× for mean entropy (0.57 vs. 0.16). This nearly four-fold improvement stems from two mechanisms. First, the
multiplicative formulation (Eq. 7) compounds spike count and variance into an amplified signal. Second, the spike detection
thresholds (τb, τr) act as a denoising filter: unlike raw entropy which fluctuates at every token, spikes are triggered only by
significant instability events—sustained entropy growth or sharp rebounds from historical minima. This sparsification yields
cleaner separation: in Case 1, spikes appear infrequently (7.0 total) against a stable background, while in Case 2, spikes
cluster densely (51.5 total) around failure regions. The result is a diagnostic signal that is both stronger in magnitude and
less cluttered by noise.
Figure 7. Token-level entropy and spike heatmaps. Left: entropy magnitude (green→red). Right: spike status (green=none, yellow=single,
orange=both). EDIS discrimination ratio (14.0×) substantially exceeds mean entropy (3.6×).
11

## Page 12

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
B. Statistical Analysis of Entropy Spikes
While EDIS uses sophisticated burst and peak-valley detection (Appendix A), we also validate that a simpler spike definition
yields statistically significant differences. We define an entropy spike as a generation step where entropy changes abruptly:
|Ht+1 −Ht| > τ for a fixed threshold τ = 0.7.
Table 3. Entropy spike counts (threshold τ = 0.7) for correct and incorrect solutions. Qwen2.5-Math-1.5B results are from RL training
validation; Qwen3-4B-Instruct results are from multi-temperature evaluation.
Model
Correct
Incorrect
Ratio
p-value
Cohen’s d
Qwen2.5-Math-1.5B
49.3 ± 23.2
82.0 ± 39.3
1.66×
< 10−100
1.03
Qwen3-4B-Instruct
36.9 ± 52.8
133.8 ± 145.6
3.62×
< 10−100
0.97
Incorrect solutions exhibit 1.7–3.6× more entropy spikes than correct ones. This difference is statistically robust (Cohen’s
d ≈1.0, p < 10−100), indicating a large and consistent effect. EDIS’s refined dual-threshold detection further amplifies
this separation to 14.0× (Appendix A), demonstrating the value of capturing sustained entropy growth and sharp rebounds
beyond simple differencing.
C. Detailed Best-of-N Scaling Results
This section presents detailed EDIS-based best-k-of-N selection results for three models, complementing the summary in
Section 5.1. Each figure shows performance across four benchmarks (rows) and three temperatures (columns), with lines
indicating average accuracy, EDIS-best, and majority voting.
C.1. Qwen2.5-Math-1.5B
Figure 8. Best-k-of-N selection for Qwen2.5-Math-1.5B. EDIS filtering more than doubles baseline accuracy on GSM8K (33.8% →
68.1%) and MATH (29.8% →56.6%).
Figure 8 shows results for the smallest model. EDIS yields dramatic improvements: GSM8K average accuracy increases
from 33.8% to 68.1%, and majority voting reaches 85% at temperature 0.6. Even at high temperature (1.0), where outputs
are noisiest, majority voting improves from 32% to 84% on GSM8K.
12

## Page 13

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
C.2. Qwen3-4B-Instruct
Figure 9. Best-k-of-N selection for Qwen3-4B-Instruct. Despite high baseline performance (> 80% on MATH), EDIS provides consistent
gains on competition benchmarks.
Figure 9 shows results for the instruction-tuned model with the highest baseline. Overall accuracy improves from 58.8% to
62.2% (+3.4 pp). While MATH is near-saturated (81–85%), EDIS provides meaningful gains on competition benchmarks:
AMC23 improves by 7.9 pp and AIME24 by 4.3 pp.
C.3. Qwen2.5-Math-7B
Figure 10 shows results for the 7B model. Overall average accuracy improves from 40.9% to 61.9% (+21.0 pp), with
majority voting reaching 69.8%. GSM8K shows the strongest gains: majority voting achieves 92% at temperature 0.6. At
high temperature, EDIS recovers performance from 28.0% to 60.0%, demonstrating robustness to noisy sampling.
C.4. Summary
Across all three models, EDIS-based filtering consistently improves accuracy as candidate pool size increases. The benefits
scale inversely with model capability: Qwen2.5-Math-1.5B gains +24.6 pp, Qwen2.5-Math-7B gains +21.0 pp, and
Qwen3-4B-Instruct gains +3.4 pp. Importantly, EDIS remains effective even for strong models, providing meaningful
improvements on challenging competition benchmarks where room for gains exists. These results confirm that entropy
trajectory stability is a robust indicator of reasoning quality across model sizes, temperatures, and problem difficulties.
D. Full Scaling Comparison Results
This section presents complete results for the comparison of selection methods (Section 5.2), broken down by sampling
temperature. Table 4 reports accuracy for each method at temperatures 0.2, 0.6, and 1.0. All experiments use Qwen2.5-
Math-1.5B.
Analysis by Temperature.
At low temperature (0.2), the model generates more deterministic outputs, resulting in lower
variance across methods. EDIS maintains a consistent advantage, particularly on GSM8K where it achieves 57–65%
accuracy compared to 44–49% for sequence entropy. At moderate temperature (0.6), EDIS shows its strongest advantage,
13

## Page 14

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
Figure 10. Best-k-of-N selection for Qwen2.5-Math-7B. EDIS filtering yields +21 pp overall improvement and enables 92% majority
voting accuracy on GSM8K at temperature 0.6.
Table 4. Selection method comparison across all temperatures. Best results per cell group in bold. Maj = Majority Voting, Ent = Sequence
Entropy, SC = Self-Certainty.
Temperature 0.2
Temperature 0.6
Temperature 1.0
Average
N
Dataset
Rand
Maj
Ent
SC
EDIS
Rand
Maj
Ent
SC
EDIS
Rand
Maj
Ent
SC
EDIS
Rand
Maj
Ent
SC
EDIS
1
GSM8K
44.38
50.00
46.00
51.00
59.00
40.12
57.00
54.00
63.00
69.00
23.88
36.00
56.00
52.00
56.00
36.12
47.67
52.00
55.33
61.33
MATH
33.00
39.00
38.00
42.00
48.00
33.00
49.00
45.00
51.00
53.00
25.12
44.00
53.00
54.00
49.00
30.38
44.00
45.33
49.00
50.00
AMC23
49.38
60.00
57.50
57.50
60.00
41.88
62.50
65.00
62.50
62.50
23.44
40.00
40.00
37.50
45.00
38.23
54.17
54.17
52.50
55.83
AIME24
8.75
13.33
16.67
16.67
20.00
8.33
16.67
16.67
16.67
20.00
3.33
6.67
6.67
10.00
3.33
6.81
12.22
13.33
14.44
14.44
Overall
36.94
43.33
41.48
44.81
50.74
34.21
50.37
48.15
53.33
56.67
21.99
36.30
47.04
45.93
45.93
31.05
43.33
45.56
48.02
51.11
2
GSM8K
43.19
48.00
44.00
51.00
57.00
39.62
60.00
50.00
63.00
73.00
23.88
43.00
65.00
57.00
69.00
35.56
50.33
53.00
57.00
66.33
MATH
32.25
43.00
39.00
43.00
46.00
31.25
45.00
43.00
48.00
61.00
20.62
39.00
59.00
57.00
60.00
28.04
42.33
47.00
49.33
55.67
AMC23
47.34
60.00
57.50
60.00
60.00
42.50
60.00
60.00
60.00
57.50
25.94
52.50
65.00
60.00
45.00
38.59
57.50
60.83
60.00
54.17
AIME24
8.54
10.00
10.00
10.00
13.33
7.50
20.00
20.00
20.00
20.00
4.17
16.67
16.67
13.33
13.33
6.74
15.56
15.56
14.44
15.56
Overall
35.90
43.70
40.37
44.81
48.52
33.38
50.00
45.56
52.22
60.37
20.79
40.00
57.41
52.59
55.93
30.02
44.57
47.78
49.88
54.94
4
GSM8K
44.59
49.00
45.00
51.00
58.00
40.19
61.00
47.00
61.00
74.00
23.19
32.00
68.00
57.00
70.00
35.99
47.33
53.33
56.33
67.33
MATH
33.59
41.00
39.00
42.00
49.00
32.91
51.00
49.00
56.00
61.00
23.38
47.00
64.00
61.00
64.00
29.96
46.33
50.67
53.00
58.00
AMC23
48.75
60.00
57.50
60.00
57.50
40.55
60.00
60.00
60.00
57.50
27.27
50.00
57.50
50.00
52.50
38.85
56.67
58.33
56.67
55.83
AIME24
8.75
13.33
13.33
13.33
16.67
7.81
20.00
20.00
16.67
23.33
4.90
16.67
23.33
20.00
23.33
7.15
16.67
18.89
16.67
21.11
Overall
37.15
43.70
41.11
44.81
50.00
33.95
52.59
46.67
54.07
61.11
21.83
38.52
60.00
53.33
60.00
30.98
44.94
49.26
50.74
57.04
8
GSM8K
44.47
51.00
49.00
55.00
65.00
41.44
63.00
55.00
70.00
76.00
22.98
35.00
66.00
51.00
76.00
36.30
49.67
56.67
58.67
72.33
MATH
33.77
42.00
40.00
43.00
47.00
31.27
49.00
47.00
55.00
68.00
23.36
47.00
64.00
59.00
65.00
29.46
46.00
50.33
52.33
60.00
AMC23
47.70
57.50
57.50
57.50
57.50
41.37
62.50
62.50
62.50
62.50
25.43
57.50
60.00
57.50
52.50
38.16
59.17
60.00
59.17
57.50
AIME24
9.17
10.00
10.00
10.00
13.33
8.59
20.00
20.00
20.00
20.00
4.79
20.00
20.00
23.33
20.00
7.52
16.67
16.67
17.78
17.78
Overall
37.06
44.07
42.59
45.93
51.48
34.01
52.96
49.26
57.78
64.81
21.46
41.11
59.26
51.85
62.22
30.84
46.05
50.37
51.85
59.51
16
GSM8K
44.04
50.00
46.00
52.00
59.00
40.55
64.00
52.00
68.00
81.00
22.07
33.00
67.00
48.00
77.00
35.55
49.00
55.00
56.00
72.33
MATH
33.48
42.00
40.00
43.00
51.00
32.41
54.00
50.00
60.00
71.00
22.65
46.00
66.00
59.00
65.00
29.51
47.33
52.00
54.00
62.33
AMC23
47.42
55.00
57.50
55.00
57.50
41.07
62.50
62.50
62.50
57.50
24.22
55.00
62.50
62.50
52.50
37.57
57.50
60.83
60.00
55.83
AIME24
8.44
13.33
13.33
13.33
13.33
7.89
23.33
23.33
23.33
26.67
4.48
16.67
23.33
20.00
26.67
6.94
17.78
20.00
18.89
22.22
Overall
36.67
43.70
41.85
44.81
50.74
33.98
55.56
49.63
59.26
67.78
20.65
39.26
61.11
51.11
63.33
30.43
46.17
50.86
51.73
60.62
14

## Page 15

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
achieving up to 81% on GSM8K at N = 16 compared to 68% for self-certainty—a gap of 13 percentage points. At high
temperature (1.0), outputs are most diverse but also noisiest. Sequence entropy becomes more competitive, matching or
exceeding EDIS in 15 out of 25 dataset-N combinations. However, EDIS maintains the best overall performance on GSM8K
(56–77%) and achieves the highest accuracy at larger candidate pools (N ≥4).
Summary.
Across all temperatures and candidate pool sizes, EDIS achieves the best or near-best overall accuracy. The
advantage is most pronounced at moderate temperature (0.6), where output diversity is balanced with quality. At high
temperature (1.0), the gap between methods narrows, but EDIS remains competitive. These results demonstrate that EDIS is
robust across different sampling configurations.
E. Stability of EDIS Across Training
A key question for practical deployment is whether EDIS requires recalibration as training progresses. If the discriminative
power of entropy dynamics varied significantly across checkpoints, practitioners would need to tune thresholds or validate
performance at each stage. We analyze this question by tracking spike ratios throughout RL training on Qwen2.5-Math-1.5B
(500 steps on NuminaMath-20K), evaluating on the AMC23 validation set every 10 steps.
Spike ratio remains stable.
Figure 11 (left) shows the ratio of mean spike counts between incorrect and correct responses
at each training checkpoint. At Step 0 (the pretrained checkpoint, before any RL fine-tuning), incorrect responses already
exhibit 1.92× more entropy spikes than correct ones. This ratio remains remarkably stable throughout training, fluctuating
between 1.90× and 2.69× with a mean of 2.26× and standard deviation of only 0.19.
Figure 11. Spike ratio stability across 500 training steps (evaluated on AMC23). Left: The ratio of spike counts (incorrect/correct)
remains stable at 1.9–2.7× throughout training, with Step 0 already showing strong discriminative power (1.92×). Right: While absolute
spike counts decrease for both correct and incorrect responses as training progresses, the relative difference persists.
Absolute counts decrease, relative difference persists.
Figure 11 (right) reveals an interesting pattern: both correct
and incorrect responses show decreasing spike counts as training progresses, reflecting the model’s increasing confidence.
However, the gap between them remains consistent—incorrect responses consistently exhibit more instability than correct
ones, regardless of the overall confidence level.
Implication for deployment.
This stability has significant practical implications:
• No checkpoint-specific calibration required: The same spike detection thresholds (τb = 1.36, τr = 1.33) that work
at Step 0 remain effective at Step 500.
• Applicable to pretrained models: EDIS provides useful signal even before any task-specific fine-tuning, enabling
immediate deployment.
15

## Page 16

EDIS: Diagnosing LLM Reasoning via Entropy Dynamics
• Robust across training dynamics: The discriminative power of EDIS is not an artifact of a particular training stage
but reflects intrinsic properties of reasoning quality.
These findings support the use of EDIS as a general-purpose diagnostic signal that does not require task-specific or
checkpoint-specific tuning.
F. Supplementary Details for RL Training
Figure 12 provides supplementary training curves for Section 5.4 at temperature T = 0.6.
Figure 12. Training dynamics at temperature 0.6. (a) Accuracy improves throughout training; EDIS-informed achieves higher peak. (b)
EDIS scores decrease as models learn; EDIS-informed reaches near-zero instability.
Accuracy.
Training improves accuracy from ∼33% to ∼48–52% over 500 steps. Notably, EDIS-informed training
achieves a slightly higher peak (52.0% vs. 48.0%), suggesting that optimizing for entropy stability complements rather than
competes with the correctness objective.
EDIS Score.
As expected, EDIS decreases throughout training for both methods—models naturally become more stable
as they improve. The key difference is magnitude: our approach reduces EDIS to near-zero (< 5), while baseline plateaus
around ∼15. This persistent gap indicates that standard training leaves residual instability that explicit EDIS-based curation
can eliminate.
Convergence.
An interesting pattern emerges in how the two methods converge. Baseline shows rapid early gains but
levels off mid-training, whereas EDIS-informed training continues improving steadily. This suggests that EDIS provides a
useful learning signal even after task accuracy has saturated.
16
