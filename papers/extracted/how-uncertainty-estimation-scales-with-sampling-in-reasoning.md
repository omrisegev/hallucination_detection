---
source_pdf: papers/How Uncertainty Estimation Scales with Sampling in Reasoning Models.pdf
slug: how-uncertainty-estimation-scales-with-sampling-in-reasoning
pages: 17
extracted_on: 2026-07-13
---

# How Uncertainty Estimation Scales with Sampling in Reasoning Models

## Page 1

How Uncertainty Estimation Scales with Sampling in Reasoning Models
Maksym Del, Markus Kängsepp, Marharyta Domnich
Ardi Tampuu, Lisa Yankovskaya, Meelis Kull, Mark Fishel
Institute of Computer Science
University of Tartu
Abstract
Uncertainty estimation is critical for deploying
reasoning language models, yet remains poorly
understood under extended chain-of-thought
reasoning. We study parallel sampling as a
fully black-box approach using verbalized con-
fidence and self-consistency. Across three rea-
soning models and 17 tasks spanning mathe-
matics, STEM, and humanities, we characterize
how these signals scale.
Both self-consistency and verbalized confi-
dence scale in reasoning models, but self-
consistency exhibits lower initial discrimina-
tion and lags behind verbalized confidence
under moderate sampling. Most uncertainty
gains, however, arise from signal combina-
tion: with just two samples, a hybrid estima-
tor improves AUROC by up to +12 on aver-
age and already outperforms either signal alone
even when scaled to much larger budgets, af-
ter which returns diminish. These effects are
domain-dependent: in mathematics, the native
domain of RLVR-style post-training, reason-
ing models achieve higher uncertainty quality
and exhibit both stronger complementarity and
faster scaling than in STEM or humanities.
1
Introduction
Uncertainty estimation is a central component of
reliable machine learning. It enables selective pre-
diction, risk-aware decision-making, and safe inte-
gration of models into larger systems. Reasoning
language models (RLMs) extend standard large
language models by extending test-time computa-
tion through extended chain-of-thought delibera-
tion. These models have demonstrated strong per-
formance and are increasingly deployed in numer-
ous domains, including those where reliable un-
certainty estimation is particularly important. Al-
though extended reasoning is primarily used to im-
prove answer accuracy, it also reveals uncertainty
information that can be extracted in a fully black-
box manner, e.g., by eliciting verbalized confidence
from a single model run (Lin et al., 2022a).
Beyond elicitation, uncertainty estimation can
also exploit parallel sampling, which queries the
same prompt multiple times to obtain multiple sam-
ples. Variation across samples can be used to esti-
mate uncertainty, for example, through consistency
in final responses (Wang et al., 2022) or by aggre-
gating introspective signals across samples (Xiong
et al., 2024). Moreover, it also becomes possible to
estimate uncertainty with a combined signal, where
introspection and consistency are added together
(Chen and Mueller, 2024).
Parallel sampling may unlock higher-quality un-
certainty estimates, but in RLMs each additional
sample is particularly costly because it entails an-
other full chain-of-thought trace. We therefore
need to understand how uncertainty estimation
scales with the number of samples, how quickly
improvements saturate, and whether combining sig-
nals yields further gains. Although uncertainty esti-
mation in reasoning language models has recently
received attention, existing studies evaluate only
the single-sample setting (Zeng et al., 2025b; Yoon
et al., 2025; Mei et al., 2025), leaving multi-sample
behavior and interactions between introspective
and agreement-based signals uncharacterized.
Why transfer of insights from standard non-
reasoning language models is not guaranteed.
In contrast, uncertainty estimation for standard
large language models has been studied in multi-
sample settings. Prior work examines agreement-
based metrics such as self-consistency, as well as
combinations of introspective and agreement-based
signals (Xiong et al., 2024; Chen and Mueller,
2024), exploring to what extent these approaches
can be effective in standard non-reasoning models.
However, a direct transfer of these findings cannot
be naively assumed due to fundamental differences
between reasoning and non-reasoning models that
1
arXiv:2603.19118v1  [cs.AI]  19 Mar 2026

## Page 2

alter how uncertainty is produced:
• First, a single sample in a reasoning model
involves extended intra-sample deliberation,
during which a larger number of candidate
solutions may be internally explored and dis-
carded. Consequently, a single sample may
already capture some uncertainty that would
only emerge through additional samples in
standard models, making the marginal value
of further sampling unclear.
• Second, as introspection is performed on
traces potentially surfacing a larger number of
hypotheses, it may change how introspective
and agreement-based signals interact. Sig-
nals that were complementary under shallow
reasoning may exhibit different synergies or
become more redundant once uncertainty is
partially resolved during deliberation.
• Third, the way uncertainty signals in reason-
ing language models scale and complement
each other can vary systematically across do-
mains compared to standard language models.
Reasoning models undergo a key post-training
procedure via reinforcement learning with ver-
ifiable rewards (RLVR), but this is primarily
limited to mathematical domains.
Research goal and scope.
Our goal is to un-
derstand how black-box uncertainty estimation be-
haves under parallel sampling in reasoning lan-
guage models. We study how introspection-based
and agreement-based uncertainty signals scale
with the number of test-time samples, how quickly
their gains saturate, and how these signals inter-
act under the extended chain-of-thought reasoning
across domains.
Contributions and Findings
• Contribution: The first systematic bench-
mark of uncertainty scaling in reasoning
language models. We present the first large-
scale empirical study of how black-box un-
certainty estimation scales with parallel sam-
pling in reasoning language models. Across
three models and 17 tasks spanning mathe-
matics, STEM, and humanities, we evaluate
verbalized confidence (VC), self-consistency
(SC), and their combination under a unified
bootstrap-based protocol, establishing a com-
prehensive empirical reference for uncertainty
estimation under extended chain-of-thought
reasoning (Section 3).
• Finding I: Self-consistency and verbalized
confidence both scale in reasoning language
models, but self-consistency lags behind.
Under parallel sampling, VC provides a strong
baseline in reasoning language models and
can be further improved via scaling, gaining
up to +10.1 AUROC in mathematics from
K=1 to K=8, but saturating earlier and yield-
ing smaller gains (+4–5 AUROC) in STEM
and humanities. In contrast, self-consistency
(SC) starts substantially below VC at compa-
rable budgets (e.g., 70.5 vs. 73.5 AUROC at
K=2 in mathematics) and, while it improves
with additional samples, does not overtake VC
within the tested range.
• Finding II: Signal complementarity domi-
nates scaling of either signal alone. Combin-
ing introspection- and agreement-based uncer-
tainty signals yields substantially larger gains
than increasing the sampling budget for VC
or SC individually. With just two samples, the
hybrid estimator (SCVC) improves AUROC
by +12.9 points in mathematics and +6.4
points in STEM and humanities relative to
single-sample VC, and already outperforms
VC or SC scaled to K=8 across all domains.
• Finding III: Uncertainty signals scale
fastest and combine most effectively in
mathematics. Scaling behavior is strongly
domain-dependent:
mathematics exhibits
both larger immediate gains from combining
signals and more sustained improvement with
additional samples (SCVC: +2.7 AUROC
from K=2→5 and +1.5 from K=5→8),
whereas STEM and humanities saturate earlier
(approximately +1–2 AUROC beyond K=2).
This aligned behavior across VC, SC, and
their combination is consistent with reason-
ing language models being most optimized
for mathematical reasoning.
2
Preliminaries
2.1
Methods
We study black-box uncertainty estimation from
parallel sampling. For each input, we sample K
independent model samples, where each sample
corresponds to a full model execution producing
2

## Page 3

a reasoning trace and a final answer {(ri, ai)}K
i=1.
A sample can additionally yield a verbalized confi-
dence score ci ∈[0, 1] if so prompted. Let ˆa denote
the majority-vote answer among {ai}K
i=1; ties are
broken uniformly at random.
Verbalized confidence (VC).
Verbalized confi-
dence uses the model’s explicit confidence outputs
as a black-box uncertainty signal (Kadavath et al.,
2022; Lin et al., 2022b). We prompt the model
to report a numeric confidence value alongside its
final answer using a standardized epistemic elicita-
tion prompt. Specifically, we adopt an Epistemic
Elicitation (EpEL; Tian et al., 2023) instruction
that encourages the model to reflect on its sub-
jective certainty in the provided answer without
revisiting the solution. The same EpEL prompt
is used consistently across all domains, including
mathematics, STEM, and humanities. We provide
complete prompts in Appendix B, and analyze the
impact of alternative confidence elicitation variants
in Section 3.5.
We rescale the reported confidence value to [0, 1]
by dividing by 100. With K sampled traces, we av-
erage confidence over those predicting the majority
answer ˆa:
VCavg =
1
|{i : ai = ˆa}|
X
i:ai=ˆa
ci.
We also refer to V Cavg simply as V C when it is
clear that it is being sampled.
Self-consistency (SC).
Self-consistency (Wang
et al., 2022) estimates confidence from agreement
across K sampled samples:
SC =
1
K
K
X
i=1
1[ai = ˆa].
Combined signal (SCVC).
SC and VC can also
be combined (Chen and Mueller, 2024; Xiong et al.,
2024; Huang et al., 2024; Rivera et al., 2024). In
this study we explore the following minimal com-
bination:
SCVC = λ · SC + (1 −λ) · VCavg,
with λ = 0.5 by default; we vary λ in Section 3.4.
2.2
Tasks and Models
Tasks.
We evaluate uncertainty estimation in
three task families: mathematics, STEM (excluding
math), and humanities, covering 17 tasks in total
(Table 1). Mathematical tasks are a key in-domain
setting for RLVR-style post-training of reasoning
models (Ma et al., 2025). Our math suite includes
GSM8K (Cobbe et al., 2021), AIME 2024 & 2025
(30 problems each, combined into a single task)
(Art of Problem Solving), and the Math subset of
MMLU-Pro (Wang et al., 2024).
For non-mathematical reasoning, we include
GPQA Diamond (Rein et al., 2024) and multiple
subject areas from MMLU-Pro (Wang et al., 2024),
spanning STEM (Health, Biology, Chemistry, Eco-
nomics, Physics, Computer Science, Engineering)
and humanities (Psychology, Law, Business, His-
tory, Philosophy, Other). Tasks are mainly multiple-
choice; math tasks additionally include free-form
numeric answers, such as GSM8K and AIME. The
number of examples per task is reported in Table 1.
Models.
We use three open-source reasoning
models: gpt-oss-20b1 (with reasoning effort set
to high), Qwen3-30B-A3B2, and DeepSeek-R1-8B3.
gpt-oss-20b and Qwen3-30B-A3B are mixture-
of-experts models trained with Reinforcement
Learning
with
Verifiable
Rewards
(RLVR).
DeepSeek-R1-8B is a dense model obtained by
fine-tuning an 8B Qwen base on reasoning traces
from DeepSeek-R1 (DeepSeek-AI et al., 2025).
We select these mid-sized models to retain strong
reasoning performance while enabling robust
parallel sampling with up to 100 samples per
question. All models support context windows
of at least 131K tokens, enabling extended
chain-of-thought reasoning.
Generation configuration.
We allow models to
generate up to 60K tokens per sample, which fits
within the 131K context window and leaves room
for an additional confidence estimation pass of
up to 60K tokens (needed for Section 3.5). All
evaluations are performed using the vLLM frame-
work4. We use generation hyperparameters recom-
mended by the model authors: temperature = 1.0
and top-p = 1.0 for gpt-oss-20b, and temperature
= 0.6 and top-p = 0.95 for Qwen3-30B-A3B and
DeepSeek-R1-8B.
1https://huggingface.co/openai/gpt-oss-20b
2https://huggingface.co/Qwen/Qwen3-30B-A3B
3https://huggingface.co/deepseek-ai/
DeepSeek-R1-0528-Qwen3-8B
4https://github.com/vllm-project/vllm
3

## Page 4

Task
# examples
Mathematical tasks
Math
1351
GSM8K
1319
AIME 2024 & 2025
60
STEM tasks (excluding math)
Health
818
Biology
717
Chemistry
1132
Economics
844
Physics
1299
Computer Science
410
Engineering
969
GPQA Diamond
198
Humanities tasks
Psychology
798
Law
1101
Business
789
History
381
Philosophy
499
Other
924
Table 1: Task families and constituent tasks used in our
evaluation, along with the number of examples per task.
2.3
AUROC for confidence evaluation
We evaluate confidence signals by how discrim-
inative they are about correctness.
A signal is
discriminative if correct answers tend to receive
higher scores than incorrect ones. A signal is cal-
ibrated if predictions with assigned confidence p
are correct about p of the time. Calibration metrics
such as ECE (Guo et al., 2017) or the Brier score
(Brier, 1950) require meaningful probabilistic in-
terpretation of the numeric scale. They depend on
scale alignment, which is undesirable here because
1) VC is self-reported and often mis-scaled (Tian
et al., 2023; Xiong et al., 2024; Chen et al.), 2) SC
is an agreement statistic with coarse, K-dependent
precision. Because both signals may be mono-
tonic but mis-calibrated, discrimination provides
a scale-invariant evaluation of uncertainty quality,
we therefore choose AUROC (Hanley and McNeil,
1982). Formally, AUROC measures the probability
that a randomly chosen correct example receives
a higher confidence score than a randomly chosen
incorrect example.
In our setting, the binary labels are answer cor-
rectness and the scores are the confidence signal
values; AUROC can be interpreted as the proba-
bility that, for a randomly chosen correct and in-
correct example, the correct one receives a higher
confidence score, with 0.5 corresponding to ran-
dom ranking and higher values indicating better
discrimination.
2.4
Bootstrap evaluation protocol
Our goal is to estimate AUROC at a target sam-
pling budget K under stochastic decoding where
the variance from temperature sampling can be
large. Specifically, for each question in our datasets,
we first generate a pool of R independent samples
(reasoning chain, answer, and confidence) using
repeated decoding. We use R = 10 for all main
results aiming to cover moderate-sampling regime
which is most practically relevant for reasoning
models where each chain of though sample is long
and expensive.. Individual generations that do not
follow the required answer or confidence format
are discarded, and questions with no valid samples
are removed (less than 1% of examples).
For each question q, we first generate a pool of
R independent samples {(rq,i, aq,i, cq,i)}R
i=1 under
stochastic decoding. To estimate performance at
sampling budget K while accounting for decoding
variance, we perform B bootstrap draws as follows.
In each draw, we uniformly sample K elements per
question without replacement, compute VC, SC, or
SCVC from these samples, and evaluate AUROC
over all questions. Task-level AUROC is macro-
averaged within domain, and model-level averages
are computed within the same draw. We report the
mean across B draws with 95% percentile intervals.
The per-question pool serves as a Monte Carlo ap-
proximation to the model’s decoding distribution,
so repeatedly drawing K elements yields plausible
datasets that one would obtain by re-decoding the
model, reducing variance on estimates while mak-
ing maximal use of the generated samples. Repeat-
ing this procedure over B bootstrap draws yields a
distribution of aggregated AUROC values.
3
Experimental Results
3.1
Finding I: SC and VC both scale in RLMs,
but SC demonstrates lower sample
efficiency in the low-budget regime
Verbalized confidence at one sample.
Verbal-
ized confidence provides strong uncertainty dis-
crimination in reasoning language models, even
with a single sample, consistent with recent find-
ings on single-sample introspection in RLMs (Zeng
et al., 2025b; Yoon et al., 2025; Mei et al., 2025)
and illustrated in Figure 1. At K=1, VC achieves
high AUROC on most tasks, including 78.8 on
MMLU-Pro Math, 68.7 on GSM8K, 73.8 on aver-
age across STEM tasks, and 68.5 across humanities.
The lower average in mathematics is driven primar-
ily by AIME (66.4 AUROC), a small and highly
sensitive benchmark, whereas larger math tasks ex-
hibit substantially stronger VC. We analyze VC
4

## Page 5

VC (K=1)
VC (K=2)
VC (K=5)
VC (K=8)
SC (K=2)
SC (K=5)
SC (K=8)
SCVC (K=2)
SCVC (K=5)
SCVC (K=8)
Mathematical tasks
Math
78.7±1.0
81.1±1.0
84.5±0.6
84.9±0.5
66.1±1.2
74.1±1.1
77.6±0.9
85.5±0.8
87.9±0.6
88.7±0.4
GSM8K
68.7±1.3
71.9±1.3
75.9±0.9
77.5±0.6
63.3±1.2
68.2±1.0
70.7±0.6
77.7±1.2
80.8±0.8
82.4±0.5
AIME 2024 & 2025
66.4±7.6
67.2±8.2
78.8±6.8
81.7±5.5
82.6±6.1
85.9±7.9
90.0±4.9
89.5±5.5
91.9±5.1
94.1±2.7
Average (tasks)
71.3±2.6
73.4±2.8
79.7±2.3
81.4±1.9
70.6±2.1
76.1±2.6
79.4±1.6
84.2±1.9
86.8±1.7
88.4±0.9
STEM tasks (excluding math)
Health
69.9±0.9
71.9±0.8
73.8±0.5
74.3±0.3
63.4±0.8
69.5±0.7
71.8±0.5
76.2±0.8
77.4±0.5
77.9±0.4
Biology
72.4±1.1
74.5±1.1
76.6±0.6
77.1±0.4
64.8±1.2
71.7±1.0
74.5±0.6
79.4±1.0
81.0±0.7
81.7±0.4
Chemistry
78.0±0.8
80.2±0.8
82.6±0.5
83.0±0.4
67.4±1.0
75.3±0.8
78.1±0.6
83.7±0.7
85.6±0.5
86.2±0.3
Economics
69.7±0.9
71.8±0.9
73.7±0.5
74.1±0.4
63.2±1.0
70.0±0.8
72.6±0.5
75.9±0.9
77.9±0.6
78.7±0.4
Physics
74.6±0.8
76.1±0.8
78.3±0.6
78.6±0.5
70.6±0.9
77.0±0.7
79.4±0.6
82.3±0.7
83.3±0.5
83.8±0.4
Computer Science
72.0±1.3
74.2±1.2
75.7±0.8
76.0±0.5
62.3±1.4
66.7±1.2
68.8±0.8
77.4±1.2
78.1±0.9
78.7±0.6
Engineering
79.2±0.6
80.9±0.6
82.5±0.4
83.0±0.3
74.6±0.8
81.9±0.5
83.5±0.4
86.5±0.5
87.0±0.4
87.3±0.3
GPQA Diamond
74.6±1.6
76.8±1.6
80.2±1.3
80.8±1.2
66.3±1.8
73.3±1.7
75.4±1.5
80.3±1.5
81.6±1.3
82.1±1.2
Average (tasks)
73.8±0.4
75.8±0.4
77.9±0.3
78.3±0.2
66.6±0.4
73.2±0.3
75.5±0.3
80.2±0.3
81.5±0.3
82.0±0.2
Humanities tasks
Psychology
68.1±1.0
70.4±0.9
72.9±0.7
73.7±0.6
63.3±0.8
70.0±0.8
72.5±0.8
75.5±0.8
77.7±0.7
78.6±0.6
Law
57.4±0.8
58.5±0.7
59.6±0.5
59.4±0.4
61.4±0.7
65.5±0.6
66.8±0.5
65.4±0.8
67.4±0.6
68.1±0.5
Business
79.2±0.8
81.2±0.8
83.0±0.5
83.2±0.3
66.3±1.1
73.7±0.8
76.3±0.5
84.3±0.7
85.6±0.5
86.1±0.3
History
63.6±1.1
65.4±1.1
67.3±0.7
67.5±0.5
60.0±1.0
64.8±0.9
66.6±0.6
69.3±1.1
70.9±0.9
71.6±0.6
Philosophy
71.7±1.0
73.7±0.9
75.5±0.6
75.9±0.4
63.5±1.0
69.9±0.8
71.7±0.6
76.9±0.9
77.8±0.7
78.2±0.5
Other
71.3±0.7
73.2±0.7
75.3±0.4
75.7±0.3
65.2±0.8
71.7±0.6
73.9±0.4
77.8±0.7
79.1±0.5
79.7±0.4
Average (tasks)
68.5±0.4
70.4±0.4
72.3±0.2
72.6±0.2
63.3±0.4
69.3±0.3
71.3±0.2
74.9±0.3
76.4±0.3
77.0±0.2
Table 2: AUROC of verbalized confidence (VC), self-consistency (SC), and their combination (SCVC) across
task families at varying sampling budgets K. Tasks are primarily from MMLU-Pro (Math, STEM, and Social
Sciences/Humanities); the STEM domain also includes GPQA Diamond, which is not part of MMLU-Pro. Rows
report per-task AUROC macro-averaged over models, and “Average (tasks)” denotes the mean ± bootstrap standard
deviation macro-averaged over models and tasks within task family (domain).
prompt sensitivity across tasks in Section 3.5 and
show that simplified elicitation substantially im-
proves VC on AIME without affecting downstream
conclusions.
Scaling verbalized confidence.
Beyond the
single-sample regime, VC can be scaled via par-
allel sampling by aggregating confidence across
samples associated with the majority answer, a set-
ting that has not been systematically characterized
for reasoning language models. VC scales sub-
stantially in mathematics, improving from 71.3 at
K=1 to 81.4 at K=8 (+10.1 AUROC), but exhibits
much smaller gains in STEM and humanities (+4.6
and +4.1, respectively). Across non-mathematical
domains, VC scaling is strongly front-loaded and
largely saturates by K≈5.
Self-consistency.
Self-consistency (SC) behaves
differently in reasoning language models than in
standard language models. At comparable sam-
pling budgets, SC starts substantially below VC
across all domains: at K=2, SC attains 70.5 AU-
ROC in mathematics, 66.6 in STEM, and 63.3 in
humanities, consistently trailing VC at the same
budget. The apparent competitiveness of SC on
AIME reflects the unusually low VC on that task;
beyond AIME, the gap is pronounced, with SC@2
reaching only 65.9 on MMLU-Pro Math and 63.3
on GSM8K, compared to 78.8 and 68.7 for VC@1
on the same benchmarks. A controlled comparison
using GPT-OSS-20B under matched token budgets
further shows that SC is markedly weaker under
extended reasoning than under shallow (LLM-like)
generation across math, physics, and psychology,
while VC remains strong and scales reliably (Fig-
ure 1). Although SC improves steadily with addi-
tional samples, it does not overtake VC within the
tested range up to K=8.
Summary.
Taken together, these results provide
the first systematic characterization of individual
uncertainty signals under parallel sampling in rea-
soning language models. Verbalized confidence
is a strong baseline that can be further improved
via sampling, particularly in mathematics, while
self-consistency emerges more slowly and remains
weaker at comparable budgets. These findings es-
tablish the individual behavior of introspection- and
agreement-based signals in RLMs, independently
of their combination.
3.2
Finding II: Signal complementarity
dominates scaling of either signal alone.
Claim.
In reasoning language models, combin-
ing introspection- and agreement-based uncertainty
signals provides substantially larger gains than in-
creasing the sampling budget for either signal alone.
While verbalized confidence and self-consistency
each benefit from additional samples, their combi-
nation unlocks most of the attainable uncertainty
5

## Page 6

0
15000
30000
45000
Output tokens per example
60
70
80
90
AUROC (%)
K=8
K=64
K=8
K=64
K=8
K=64
SC
0
15000
30000
45000
Output tokens per example
K=8
K=64
K=8
K=64
K=8
K=64
VC
0
15000
30000
45000
Output tokens per example
K=8
K=64
K=8
K=64
K=8
K=64
SCVC
MMLU-Pro Math
MMLU-Pro Physics
MMLU-Pro Psychology
Thinking
Non-thinking
Figure 1: Direct comparison (AUROC vs. cost across datasets) between extended thinking (gpt-oss-20b-high) and
shallow thinking (gpt-oss-20b-low).
quality at the smallest sampling budget where both
signals become available.
Evidence.
Table 2 shows that combining verbal-
ized confidence and self-consistency yields large,
strongly front-loaded improvements across all do-
mains. In mathematics, SCVC at K=2 reaches an
average AUROC of 84.2, improving over single-
sample VC by +12.9 points (71.3→84.2) and ex-
ceeding the best performance of either VC or SC
even when scaled to K=8 (81.4 and 79.6, respec-
tively), with gains far exceeding bootstrap variabil-
ity. In STEM and humanities, SCVC estimated
with two samples improves over single-sample VC
by +6.4 AUROC (73.8→80.2 and 68.5→74.9) and
surpasses the strongest single-signal estimates at
K=8 by +1.8 and +2.3 AUROC, respectively.
Further scaling of SCVC exhibits diminishing re-
turns, but remains feasible, particularly for Math:
adding six more samples yields a +4.2 AUROC
gain on Math and roughly +2 AUROC on STEM
and Humanities.
Context.
Hybrid uncertainty estimators that com-
bine introspective confidence and cross-sample
agreement have been explored previously in stan-
dard language models (Chen and Mueller, 2023;
Xiong et al., 2024; Rivera et al., 2024; Huang et al.,
2024). However, prior work did not systematically
isolate the contribution of signal complementarity
relative to increased sampling depth. This distinc-
tion was less relevant in standard language models,
where per-sample token cost is lower and sampling
is substantially more affordable than in RLMs.
Moreover, this behavior may have not been tak-
ing place under extended chain-of-thought reason-
ing: a single reasoning trace could already inte-
grate over multiple hypotheses, potentially reduc-
ing the marginal value of cross-sample agreement
(Podolak and Verma, 2025). Our results show that
this is not the case: introspection- and agreement-
based signals remain strongly complementary even
after extended deliberation.
Implication.
As a result, the practical signifi-
cance of the low-sample complementarity finding
is amplified in reasoning language models, where
each additional sample entails a long and costly
chain-of-thought trace. Rather than allocating com-
pute to deeper sampling of a single uncertainty
signal, our results indicate that drawing a single
additional sample and combining verbalized confi-
dence with self-consistency yields the largest im-
provement per unit cost. This leads to a simple and
robust recipe for uncertainty estimation in RLMs:
avoid single-sample estimation, avoid pure self-
consistency, and instead combine introspective con-
fidence and cross-sample agreement using two sam-
ples.
3.3
Finding III: Uncertainty signals scale
fastest and combine most effectively in
mathematics
Uncertainty estimation exhibits strong domain de-
pendence in both scaling speed and signal comple-
6

## Page 7

0.0
0.2
0.4
0.6
0.8
1.0
lambda (SC weight)
0.6
0.7
0.8
0.9
AUROC
K = 2
0.0
0.2
0.4
0.6
0.8
1.0
lambda (SC weight)
K = 5
0.0
0.2
0.4
0.6
0.8
1.0
lambda (SC weight)
K = 8
Math
STEM
Humanities
Figure 2: AUROC as a function of the SC weight λ in
the hybrid SC+VC signal, shown for K=2, 5, and 8. Re-
sults are averaged across models and tasks within each
domain, with shaded regions indicating 95% confidence
intervals. Performance is stable across a wide range of
λ, with degradation only at the extremes corresponding
to pure VC or pure SC.
mentarity. Mathematics stands out as the domain
where uncertainty signals improve most rapidly
and combine most effectively. Moving from single-
sample VC to the hybrid SCVC estimator at K=2
yields a large gain in mathematics (+12.9 AUROC),
after which SCVC continues to improve from K=2
to K=5 (+2.7) and still gains from K=5 to K=8
(+1.5). In contrast, STEM and humanities show
smaller initial gains (+6.4 AUROC) and much ear-
lier saturation, with only ≈+1–2 AUROC improve-
ment beyond K=2.
This pattern mirrors the behavior of individual
signals: VC scales more strongly in mathematics,
and SC reaches higher absolute quality and exhibits
stronger late-stage scaling than in other domains.
The alignment of faster scaling for VC, SC, and
their combination indicates that uncertainty signals
are both richer and more complementary in mathe-
matical reasoning. While we do not isolate training
causes, this behavior is consistent with reasoning
language models being most effective and most
extensively optimized in mathematics.
3.4
Analysis
Combination of SC and VC is robust to weight-
ing parameter λ
So far, we combine VC and SC
using an equal-weighted sum (λ=0.5). We test the
sensitivity of this hybrid to the weighting parame-
ter λ ∈[0, 1], where λ=0 and λ=1 correspond to
pure VC and pure SC, respectively.
Figure 2 shows AUROC as a function of λ for
K ∈{2, 5, 8}, averaged across models and tasks
within each domain. Across all domains and sam-
pling budgets, SC+VC performance is largely in-
variant to λ over a wide interior range: any non-
degenerate combination (0 < λ < 1) yields nearly
2
3
4
5
8
K (number of samples)
0.18
0.24
0.30
0.36
Kendall tau (SC vs VC)
Math
STEM
Humanities
Figure 3: Kendall’s τ rank correlation between VC and
SC as a function of the number of samples K macro-
averaged across reasoning models and task families.
Correlation starts low and increases with sampling depth
mirroring describing front-loaded gains of simple ad-
dition of the two signals and is consistently lower in
mathematics than in STEM and humanities coinciding
with RLVR training on math.
identical AUROC, with degradation only at the ex-
tremes where one signal is removed.
This robustness indicates that hybrid gains do
not rely on precise weighting, but rather on the
presence of both signals, making simple equal-
weighted combination sufficient in practice.
Complementarity can be described by correla-
tion
We analyze the relationship between verbal-
ized confidence and self-consistency by measuring
their rank correlation as a function of sampling bud-
get K and domain. Figure 3 reports Kendall’s τ
between VC and SC. Across all domains, the cor-
relation increases monotonically with the number
of samples, indicating that the two signals become
progressively more aligned as additional samples
are drawn. Moreover, the correlation is consistently
higher in non-mathematical domains than in math-
ematics, particularly at small sampling budgets.
These regimes coincide with the behavior ob-
served in Section 3.2: hybrid gains are largest when
correlation is lowest (early sampling and mathemat-
ics), and diminish as correlation increases (deeper
sampling and non-RLVR-aligned domains). This
shows that the benefit of hybrid uncertainty esti-
mation does not arise from persistent signal inde-
pendence. Instead, complementarity is strongest
when VC and SC capture transiently distinct uncer-
tainty information early in sampling, and weakens
as both signals converge toward a shared notion of
uncertainty with deeper sampling or outside RLVR-
aligned domains.
7

## Page 8

3.5
Revisiting verbalized confidence variants
in reasoning language models
We study two families of verbalized confidence
(VC) methods: elicitation (Xiong et al., 2024),
where the model reports confidence alongside its
answer, and judge approaches (Gu et al., 2025),
where a separate pass reads the full reasoning trace
and outputs a confidence score. Both estimate un-
certainty via VC but differ in how the signal is ex-
tracted. We evaluate three instruction variants, each
mapping responses to a 1–100 confidence scale
(Appendix B), yielding six VC methods that incen-
tivize different forms of introspection in reasoning
language models (RLMs). While advanced vari-
ants showed limited success in short-trace LLMs,
the longer chains of thought in RLMs may better
support their assumptions, motivating a systematic
comparison.
Vanilla elicitation (VaEl). The model provides
an answer and a confidence score. This assumes
direct introspective access to uncertainty and has
been shown to work well in both LLMs and RLMs
(Xiong et al., 2024; Yoon et al., 2025; Zeng et al.,
2025a).
Verification elicitation (VeEl). The model is
prompted to check its reasoning before assigning
confidence. While short LLM traces often lack
structure, RLM scratchpads may better support self-
verification (Miao et al., 2023).
Epistemic elicitation (EpEl).
The model is
steered to reflect on its certainty during reasoning.
This relies on extended reasoning budgets enabling
simultaneous problem solving and self-assessment.
Prior work found little benefit in LLMs (Tian et al.,
2023), but longer RLM traces make this assump-
tion more plausible.
Vanilla judge (VaJu). A second pass reads the
full reasoning trace and outputs a confidence score.
Sparse traces limited this signal in LLMs, while
RLMs provide richer evidence (Xiong et al., 2024).
Verification judge (VeJu). The judge evaluates
the validity and consistency of reasoning steps.
This depends on explicit logical structure, which
RLM traces expose more clearly than LLM outputs
(Miao et al., 2023).
Epistemic-markers judge (EpJu). The judge at-
tends to hedging language and certainty cues. Such
markers were unreliable in LLMs (Liu et al., 2025)
but occur more frequently in extended RLM traces
(Venhoff et al., 2025).
Method
Math
STEM
Humanities
VaEl
73.24
77.15
70.04
VeEl
70.80
76.95
70.61
EpEl
69.44
76.11
71.43
VaJu
70.35
76.24
71.78
VeJu
73.82
76.37
71.49
EpJu
67.59
72.00
67.97
Table 3: Domain-averaged AUROC for six uncertainty
elicitation methods. Math includes AIME, GSM8K,
and MMLU-Pro Math; STEM includes Physics, Health,
and GPQA; Humanities includes remaining MMLU-Pro
tasks.
Result.
This analysis uses the same three rea-
soning models as earlier sections; STEM includes
GPQA and MMLU-Pro Physics and Health, and
humanities include MMLU-Pro Psychology, Busi-
ness, and Law.
Table 3 shows that in mathematics, minimally
guided methods perform best, with vanilla elicita-
tion (VaEl) and verification judging (VeJu) achiev-
ing 73.2 and 73.8 AUROC, while epistemic vari-
ants underperform. In STEM, elicitation domi-
nates, with VaEl highest at 77.2 AUROC. In hu-
manities, differences narrow and judgment-based
variants slightly outperform elicitation-based ones.
Epistemic-marker judging is consistently weakest.
Judge differences arise despite identical reason-
ing traces, indicating sensitivity to confidence elic-
itation rather than information availability. More-
over, SCVC with two samples delivers substantially
larger gains than judge-based methods while avoid-
ing an extra reasoning-model pass, making judges
an unfavorable cost–benefit tradeoff in RLMs.
4
Conclusion
We studied how uncertainty estimation scaled with
parallel sampling in reasoning LMs using black-
box methods.
Both verbalized confidence and
self-consistency improved with sampling, but self-
consistency consistently lagged. Most gains arose
from signal complementarity: combining both sig-
nals with just two samples outperformed deeper
sampling of either alone. Effects were domain-
dependent, with the strongest gains in mathematics,
while advanced VC variants offered little or no ben-
efit. These insights provide practical guidance for
sampling-based uncertainty estimation in reasoning
models, where extra samples are costly.
8

## Page 9

References
Art of Problem Solving. Aime problems and solutions.
https://artofproblemsolving.com/wiki/
index.php/AIME_Problems_and_Solutions.
Glenn W. Brier. 1950.
Verification of forecasts ex-
pressed in terms of probability. Monthly Weather
Review, 78(1):1–3.
Jiuhai Chen and Jonas Mueller. 2023.
Quantifying
uncertainty in answers from any language model
and enhancing their trustworthiness. arXiv preprint
arXiv:2308.16175.
Jiuhai Chen and Jonas Mueller. 2024.
Quantifying
uncertainty in answers from any language model
and enhancing their trustworthiness. In Proceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 5186–5200, Bangkok, Thailand. Association
for Computational Linguistics.
Yanda Chen,
Joe Benton,
Ansh Radhakrishnan,
Jonathan Uesato, Carson Denison, John Schulman,
Arushi Somani, Peter Hase, Misha Wagner, Fabien
Roger, Vlad Mikulik, Sam Bowman, Jan Leike, Jared
Kaplan, and Ethan Perez. Reasoning Models Don’t
Always Say What They Think.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems. Preprint, arXiv:2110.14168.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,
Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong
Shao, Zhuoshu Li, Ziyi Gao, and 181 others. 2025.
Deepseek-r1 incentivizes reasoning in llms through
reinforcement learning. Nature, 645:633–638.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,
Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun
Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni,
and Jian Guo. 2025. A survey on llm-as-a-judge.
Preprint, arXiv:2411.15594.
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Wein-
berger. 2017. On calibration of modern neural net-
works. Proceedings of Machine Learning Research,
70:1321–1330.
James A Hanley and Barbara J McNeil. 1982. The
meaning and use of the area under a receiver operat-
ing characteristic (roc) curve. Radiology, 143(1):29–
36.
Yukun Huang, Yixin Liu, Raghuveer Thirukovalluru,
Arman Cohan, and Bhuwan Dhingra. 2024. Cali-
brating long-form generations from large language
models. Preprint, arXiv:2402.06544.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli
Tran-Johnson, Scott Johnston, Sheer El-Showk,
Andy Jones, Nelson Elhage, Tristan Hume, Anna
Chen,
Yuntao Bai,
Sam Bowman,
Stanislav
Fort, and 17 others. 2022.
Language Models
(Mostly) Know What They Know. arXiv preprint.
ArXiv:2207.05221 [cs].
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022a.
Teaching models to express their uncertainty in
words. Transactions on Machine Learning Research.
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022b.
Teaching models to express their uncertainty in
words. arXiv preprint arXiv:2205.14334.
Jiayu Liu, Qing Zong, Weiqi Wang, and Yangqiu
Song. 2025. Revisiting epistemic markers in con-
fidence estimation: Can markers accurately reflect
large language models’ uncertainty?
Preprint,
arXiv:2505.24778.
Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang,
Zejun Ma, and Wenhu Chen. 2025.
General-
reasoner: Advancing llm reasoning across all do-
mains. Preprint, arXiv:2505.14652.
Zhiting Mei, Christina Zhang, Tenny Yin, Justin Lidard,
Ola Shorinwa, and Anirudha Majumdar. 2025. Rea-
soning about uncertainty: Do reasoning models know
when they don’t know? Preprint, arXiv:2506.18183.
Ning Miao, Yee Whye Teh, and Tom Rainforth. 2023.
Selfcheck: Using llms to zero-shot check their own
step-by-step reasoning. Preprint, arXiv:2308.00436.
Jakub Podolak and Rajeev Verma. 2025. Read your
own mind: Reasoning helps surface self-confidence
signals in llms. Preprint, arXiv:2505.23845.
David Rein, Betty Li Hou, Asa Cooper Stickland, Jack-
son Petty, Richard Yuanzhe Pang, Julien Dirani, Ju-
lian Michael, and Samuel R. Bowman. 2024. GPQA:
A graduate-level google-proof q&a benchmark. In
First Conference on Language Modeling.
Mauricio Rivera, Jean-François Godbout, Reihaneh
Rabbany, and Kellin Pelrine. 2024. Combining confi-
dence elicitation and sample-based methods for un-
certainty quantification in misinformation mitigation.
arXiv preprint arXiv:2401.08694.
Katherine Tian, Eric Mitchell, Allan Zhou, Archit
Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn,
and Christopher D. Manning. 2023. Just Ask for
Calibration: Strategies for Eliciting Calibrated Con-
fidence Scores from Language Models Fine-Tuned
with Human Feedback. Publisher: arXiv Version
Number: 2.
Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur
Conmy, and Neel Nanda. 2025. Understanding rea-
soning in thinking language models via steering vec-
tors. Preprint, arXiv:2506.18167.
9

## Page 10

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le,
Ed Chi, Sharan Narang, Aakanksha Chowdhery, and
Denny Zhou. 2022. Self-consistency improves chain
of thought reasoning in language models.
arXiv
preprint arXiv:2203.11171.
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni,
Abhranil Chandra, Shiguang Guo, Weiming Ren,
Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max
Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang
Yue, and Wenhu Chen. 2024. MMLU-pro: A more
robust and challenging multi-task language under-
standing benchmark. In The Thirty-eight Conference
on Neural Information Processing Systems Datasets
and Benchmarks Track.
Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie
Fu, Junxian He, and Bryan Hooi. 2024. Can LLMs
Express Their Uncertainty? An Empirical Evaluation
of Confidence Elicitation in LLMs. arXiv preprint.
ArXiv:2306.13063 [cs].
Dongkeun Yoon, Seungone Kim, Sohee Yang, Sunky-
oung Kim, Soyeon Kim, Yongil Kim, Eunbi Choi,
Yireun Kim, and Minjoon Seo. 2025. Reasoning
models better express their confidence. In The Thirty-
ninth Annual Conference on Neural Information Pro-
cessing Systems.
Qingcheng Zeng, Weihao Xuan, Leyang Cui, and
Rob Voigt. 2025a.
Do Reasoning Models Show
Better Verbalized Calibration?
arXiv preprint.
ArXiv:2504.06564 [cs].
Qingcheng Zeng, Weihao Xuan, Leyang Cui, and Rob
Voigt. 2025b. Thinking out loud: Do reasoning mod-
els know when they’re right?
In Proceedings of
the 2025 Conference on Empirical Methods in Natu-
ral Language Processing, pages 1394–1407, Suzhou,
China. Association for Computational Linguistics.
A
Appendix: Breakdown of AUROC and
Accuracy values per tasks and models
Tables 4–6 report the AUROC values for verbalized
confidence, self-consistency, and their combina-
tion across multiple domains. Each table presents
results stratified by model, task, and number of
samples. Specifically, Table 4 focuses on mathe-
matics tasks, Table 5 on STEM tasks, and Table 6
on humanities tasks.
Tables 7–9 present the corresponding results in
terms of accuracy, following the same structure and
domain breakdown.
10

## Page 11

VC (K=1)
VC (K=2)
VC (K=5)
VC (K=8)
SC (K=2)
SC (K=5)
SC (K=8)
SCVC (K=2)
SCVC (K=5)
SCVC (K=8)
Math
gpt-oss-20b
80.9±2.1
83.2±1.9
87.9±1.0
88.6±0.7
63.6±2.1
71.5±2.0
75.2±1.5
87.1±1.5
89.4±0.9
90.0±0.8
Qwen3-30B-A3B
76.3±1.7
79.4±1.5
81.8±1.0
81.6±0.7
62.8±2.1
70.6±1.9
74.1±1.7
82.2±1.5
85.2±1.1
86.5±0.8
DeepSeek-R1-8B
78.9±1.8
80.7±1.7
83.8±1.2
84.5±0.9
71.8±2.0
80.1±1.6
83.4±1.4
87.2±1.3
89.0±1.0
89.8±0.7
Average (models)
78.7±1.0
81.1±1.0
84.5±0.6
84.9±0.5
66.1±1.2
74.1±1.1
77.6±0.9
85.5±0.8
87.9±0.6
88.7±0.4
GSM8K
gpt-oss-20b
69.6±2.3
72.2±2.3
75.7±1.5
76.9±1.1
65.8±2.3
72.6±1.8
75.9±1.2
79.6±2.1
82.7±1.4
84.3±0.9
Qwen3-30B-A3B
67.1±2.5
70.8±2.4
75.6±1.7
77.7±1.0
57.0±1.9
58.6±1.6
59.7±0.9
73.7±2.2
76.4±1.6
78.0±1.0
DeepSeek-R1-8B
69.5±1.9
72.6±1.9
76.4±1.4
78.0±0.9
67.1±2.1
73.4±1.7
76.7±1.1
79.7±1.8
83.1±1.3
84.8±0.8
Average (models)
68.7±1.3
71.9±1.3
75.9±0.9
77.5±0.6
63.3±1.2
68.2±1.0
70.7±0.6
77.7±1.2
80.8±0.8
82.4±0.5
AIME 2024 & 2025
gpt-oss-20b
57.6±21.1
56.2±23.2
76.4±17.7
79.3±13.9
81.4±16.7
82.0±22.8
92.4±13.6
88.1±15.0
89.3±14.5
95.1±7.1
Qwen3-30B-A3B
56.3±6.3
59.9±7.6
67.4±9.1
70.6±9.7
82.5±6.3
87.0±4.8
88.0±4.7
85.6±6.3
89.5±4.2
90.1±3.9
DeepSeek-R1-8B
85.3±5.0
85.6±5.2
92.7±4.6
95.2±3.7
83.8±4.4
88.8±3.5
89.7±3.2
94.8±2.6
96.8±1.7
97.1±1.4
Average (models)
66.4±7.6
67.2±8.2
78.8±6.8
81.7±5.5
82.6±6.1
85.9±7.9
90.0±4.9
89.5±5.5
91.9±5.1
94.1±2.7
Average (tasks)
gpt-oss-20b
69.4±7.1
70.5±7.8
80.0±5.9
81.6±4.7
70.3±5.6
75.4±7.6
81.2±4.6
84.9±5.0
87.1±4.8
89.8±2.4
Qwen3-30B-A3B
66.5±2.3
70.1±2.7
74.9±3.1
76.6±3.3
67.4±2.3
72.1±1.8
73.9±1.7
80.5±2.3
83.7±1.6
84.9±1.4
DeepSeek-R1-8B
77.9±1.9
79.6±1.9
84.3±1.7
85.9±1.3
74.2±1.8
80.8±1.4
83.3±1.2
87.2±1.2
89.6±0.8
90.6±0.6
Average (models)
71.3±2.6
73.4±2.8
79.7±2.3
81.4±1.9
70.6±2.1
76.1±2.6
79.4±1.6
84.2±1.9
86.8±1.7
88.4±0.9
Table 4: AUROC of verbalized confidence (VC), self-consistency (SC), and their combination (SCVC) across
mathematics tasks at different sampling budgets K. Rows report per-model results, with the “Average (models)”
row denoting the mean ± bootstrap standard deviation after averaging across models.
VC (K=1)
VC (K=2)
VC (K=5)
VC (K=8)
SC (K=2)
SC (K=5)
SC (K=8)
SCVC (K=2)
SCVC (K=5)
SCVC (K=8)
Health
gpt-oss-20b
74.3±1.4
76.4±1.3
78.6±0.8
79.1±0.6
64.4±1.5
71.2±1.3
73.9±0.9
79.8±1.2
80.3±0.9
80.6±0.7
Qwen3-30B-A3B
67.4±1.6
69.7±1.4
72.8±0.9
73.3±0.6
62.1±1.5
68.7±1.1
71.0±0.8
74.3±1.4
76.7±0.9
77.4±0.7
DeepSeek-R1-8B
67.9±1.3
69.7±1.2
70.1±0.8
70.4±0.6
63.6±1.3
68.5±1.2
70.4±0.9
74.6±1.2
75.2±0.9
75.6±0.7
Average (models)
69.9±0.9
71.9±0.8
73.8±0.5
74.3±0.3
63.4±0.8
69.5±0.7
71.8±0.5
76.2±0.8
77.4±0.5
77.9±0.4
Biology
gpt-oss-20b
76.3±1.9
78.2±1.8
80.6±1.1
81.0±0.7
68.1±2.3
76.9±2.0
80.6±1.3
83.4±1.6
84.7±1.2
85.4±0.9
Qwen3-30B-A3B
67.4±2.1
69.7±2.0
73.1±1.2
74.0±0.6
59.9±2.0
65.3±1.6
67.6±0.9
74.3±1.9
76.5±1.2
77.2±0.6
DeepSeek-R1-8B
73.5±1.9
75.5±1.8
76.1±1.1
76.3±0.7
66.4±2.0
73.0±1.6
75.4±1.0
80.6±1.6
81.7±1.1
82.4±0.7
Average (models)
72.4±1.1
74.5±1.1
76.6±0.6
77.1±0.4
64.8±1.2
71.7±1.0
74.5±0.6
79.4±1.0
81.0±0.7
81.7±0.4
Chemistry
gpt-oss-20b
80.7±1.5
82.8±1.3
84.9±0.8
85.4±0.6
67.1±1.7
74.6±1.4
77.4±1.2
85.7±1.1
86.9±0.9
87.5±0.7
Qwen3-30B-A3B
72.7±1.4
75.1±1.3
78.8±0.9
78.7±0.7
63.4±1.7
72.5±1.3
75.9±0.9
78.9±1.3
82.6±0.9
83.7±0.6
DeepSeek-R1-8B
80.6±1.3
82.7±1.2
84.2±0.8
84.7±0.5
71.6±1.7
78.7±1.2
81.0±0.8
86.6±1.1
87.2±0.8
87.5±0.5
Average (models)
78.0±0.8
80.2±0.8
82.6±0.5
83.0±0.4
67.4±1.0
75.3±0.8
78.1±0.6
83.7±0.7
85.6±0.5
86.2±0.3
Economics
gpt-oss-20b
73.3±1.6
75.3±1.5
76.9±0.8
77.4±0.5
64.3±1.7
70.7±1.4
73.7±1.0
79.3±1.4
80.4±1.0
81.2±0.7
Qwen3-30B-A3B
66.8±1.7
69.7±1.6
72.9±0.9
73.3±0.6
60.2±1.6
67.5±1.4
70.1±0.9
73.3±1.6
76.7±1.0
77.8±0.7
DeepSeek-R1-8B
68.9±1.7
70.5±1.5
71.4±1.0
71.5±0.6
65.2±1.6
71.9±1.3
74.0±0.9
75.2±1.5
76.7±1.0
77.2±0.7
Average (models)
69.7±0.9
71.8±0.9
73.7±0.5
74.1±0.4
63.2±1.0
70.0±0.8
72.6±0.5
75.9±0.9
77.9±0.6
78.7±0.4
Physics
gpt-oss-20b
76.4±1.5
77.2±1.4
79.3±1.2
79.8±1.1
76.6±1.3
81.7±1.1
83.5±1.0
85.4±1.0
85.2±0.9
85.7±0.8
Qwen3-30B-A3B
72.2±1.4
74.4±1.3
77.8±0.9
77.8±0.8
64.7±1.7
72.3±1.5
75.1±1.2
79.0±1.3
81.3±0.9
82.1±0.8
DeepSeek-R1-8B
75.2±1.3
76.8±1.3
77.9±0.9
78.2±0.7
70.5±1.5
77.0±1.3
79.4±1.0
82.5±1.1
83.2±0.9
83.7±0.7
Average (models)
74.6±0.8
76.1±0.8
78.3±0.6
78.6±0.5
70.6±0.9
77.0±0.7
79.4±0.6
82.3±0.7
83.3±0.5
83.8±0.4
Computer Science
gpt-oss-20b
74.4±2.4
76.1±2.1
76.9±1.4
77.5±0.9
62.3±2.7
66.5±2.2
68.9±1.6
78.9±2.0
78.2±1.4
78.7±1.1
Qwen3-30B-A3B
68.7±2.3
71.3±2.1
73.2±1.5
73.2±1.0
60.3±2.4
66.0±2.0
68.4±1.3
74.6±2.3
77.6±1.5
79.0±0.9
DeepSeek-R1-8B
73.0±2.2
75.0±2.0
77.0±1.4
77.3±0.9
64.1±2.2
67.5±2.1
68.9±1.5
78.7±1.9
78.5±1.4
78.3±1.1
Average (models)
72.0±1.3
74.2±1.2
75.7±0.8
76.0±0.5
62.3±1.4
66.7±1.2
68.8±0.8
77.4±1.2
78.1±0.9
78.7±0.6
Engineering
gpt-oss-20b
82.5±1.1
83.9±1.0
84.6±0.6
84.9±0.4
74.3±1.3
81.7±0.9
83.4±0.6
87.4±0.8
87.5±0.6
87.8±0.5
Qwen3-30B-A3B
76.6±1.2
78.7±1.0
81.8±0.7
82.0±0.5
72.8±1.4
80.8±1.0
82.5±0.7
85.6±0.9
86.7±0.7
86.8±0.5
DeepSeek-R1-8B
78.6±1.0
80.1±1.0
81.3±0.7
81.9±0.5
76.7±1.3
83.2±1.0
84.7±0.6
86.6±0.9
86.8±0.7
87.2±0.5
Average (models)
79.2±0.6
80.9±0.6
82.5±0.4
83.0±0.3
74.6±0.8
81.9±0.5
83.5±0.4
86.5±0.5
87.0±0.4
87.3±0.3
GPQA Diamond
gpt-oss-20b
78.1±3.0
79.6±2.8
81.6±2.3
82.0±2.1
68.6±3.2
74.3±2.8
75.6±2.7
82.2±2.5
81.5±2.1
81.3±2.2
Qwen3-30B-A3B
68.1±3.1
71.1±3.0
76.3±2.2
76.9±2.1
64.1±3.1
72.8±3.0
75.9±2.6
77.0±2.9
80.6±2.4
81.6±2.0
DeepSeek-R1-8B
77.7±2.6
79.8±2.5
82.8±2.0
83.6±1.8
66.2±3.0
72.8±2.7
74.7±2.5
81.6±2.5
82.8±2.1
83.3±2.0
Average (models)
74.6±1.6
76.8±1.6
80.2±1.3
80.8±1.2
66.3±1.8
73.3±1.7
75.4±1.5
80.3±1.5
81.6±1.3
82.1±1.2
Average (tasks)
gpt-oss-20b
77.0±0.7
78.7±0.6
80.4±0.4
80.9±0.4
68.2±0.7
74.7±0.6
77.1±0.5
82.8±0.5
83.1±0.4
83.5±0.4
Qwen3-30B-A3B
70.0±0.7
72.5±0.6
75.8±0.4
76.2±0.3
63.4±0.7
70.7±0.6
73.3±0.5
77.1±0.6
79.8±0.5
80.7±0.3
DeepSeek-R1-8B
74.4±0.6
76.3±0.6
77.6±0.4
78.0±0.3
68.1±0.7
74.1±0.6
76.1±0.4
80.8±0.5
81.5±0.4
81.9±0.3
Average (models)
73.8±0.4
75.8±0.4
77.9±0.3
78.3±0.2
66.6±0.4
73.2±0.3
75.5±0.3
80.2±0.3
81.5±0.3
82.0±0.2
Table 5: AUROC of verbalized confidence (VC), self-consistency (SC), and their combination (SCVC) across
STEM tasks at different sampling budgets K. Rows report per-model results, with the “Average (models)” row
denoting the mean ± bootstrap standard deviation after averaging across models..
11

## Page 12

VC (K=1)
VC (K=2)
VC (K=5)
VC (K=8)
SC (K=2)
SC (K=5)
SC (K=8)
SCVC (K=2)
SCVC (K=5)
SCVC (K=8)
Psychology
gpt-oss-20b
70.0±1.7
72.3±1.7
75.9±1.2
77.0±1.1
66.0±1.4
73.2±1.5
75.8±1.3
78.2±1.4
80.0±1.1
80.8±1.0
Qwen3-30B-A3B
65.2±1.8
68.1±1.6
71.3±1.3
72.1±1.1
60.6±1.5
67.3±1.5
69.8±1.3
72.9±1.6
76.7±1.3
77.9±1.1
DeepSeek-R1-8B
68.9±1.5
70.8±1.4
71.6±1.0
71.9±0.9
63.2±1.5
69.5±1.5
71.9±1.4
75.3±1.3
76.5±1.1
77.1±1.0
Average (models)
68.1±1.0
70.4±0.9
72.9±0.7
73.7±0.6
63.3±0.8
70.0±0.8
72.5±0.8
75.5±0.8
77.7±0.7
78.6±0.6
Law
gpt-oss-20b
58.4±1.4
59.7±1.4
61.0±1.0
61.6±0.7
61.8±1.3
65.5±1.1
66.6±0.8
66.3±1.4
67.6±1.1
67.9±0.8
Qwen3-30B-A3B
56.5±1.1
57.6±1.2
59.8±0.9
59.1±0.6
60.4±1.1
64.4±1.0
65.4±0.7
64.5±1.3
67.0±1.0
67.5±0.7
DeepSeek-R1-8B
57.2±1.4
58.2±1.3
57.9±0.9
57.6±0.7
61.9±1.3
66.6±1.1
68.4±0.9
65.4±1.4
67.7±1.1
68.8±0.9
Average (models)
57.4±0.8
58.5±0.7
59.6±0.5
59.4±0.4
61.4±0.7
65.5±0.6
66.8±0.5
65.4±0.8
67.4±0.6
68.1±0.5
Business
gpt-oss-20b
82.5±1.5
84.2±1.3
86.2±0.7
86.5±0.5
67.6±1.9
76.1±1.4
78.9±0.9
87.2±1.2
88.9±0.8
89.4±0.5
Qwen3-30B-A3B
76.1±1.5
78.6±1.3
81.0±0.8
81.4±0.4
62.9±1.7
69.5±1.3
71.8±0.8
81.7±1.3
83.4±0.9
84.1±0.5
DeepSeek-R1-8B
79.0±1.4
80.9±1.3
81.7±0.8
81.8±0.5
68.5±1.8
75.4±1.5
78.1±0.9
84.1±1.2
84.5±0.9
84.9±0.6
Average (models)
79.2±0.8
81.2±0.8
83.0±0.5
83.2±0.3
66.3±1.1
73.7±0.8
76.3±0.5
84.3±0.7
85.6±0.5
86.1±0.3
History
gpt-oss-20b
65.2±2.1
66.9±1.9
68.6±1.2
69.2±0.8
62.1±1.9
67.7±1.8
70.1±1.2
71.2±1.9
72.3±1.5
73.1±1.1
Qwen3-30B-A3B
61.2±1.9
63.4±1.9
65.9±1.3
65.7±0.8
58.4±1.7
62.7±1.5
64.4±1.0
67.8±2.0
70.6±1.5
71.5±0.9
DeepSeek-R1-8B
64.3±2.1
66.0±1.9
67.3±1.2
67.6±0.8
59.6±1.8
63.9±1.7
65.4±1.2
68.9±1.9
69.8±1.5
70.1±1.1
Average (models)
63.6±1.1
65.4±1.1
67.3±0.7
67.5±0.5
60.0±1.0
64.8±0.9
66.6±0.6
69.3±1.1
70.9±0.9
71.6±0.6
Philosophy
gpt-oss-20b
77.6±1.5
79.5±1.4
80.5±0.9
80.9±0.5
65.6±1.7
72.0±1.4
73.7±1.0
81.4±1.4
80.9±1.1
80.9±0.8
Qwen3-30B-A3B
64.2±1.9
66.5±1.8
68.5±1.2
68.9±0.7
61.2±1.6
66.8±1.4
68.7±1.0
71.4±1.7
73.3±1.3
73.9±0.9
DeepSeek-R1-8B
73.3±1.6
75.2±1.4
77.4±1.0
77.8±0.7
63.8±1.7
70.8±1.4
72.8±1.0
77.9±1.5
79.2±1.2
79.7±0.8
Average (models)
71.7±1.0
73.7±0.9
75.5±0.6
75.9±0.4
63.5±1.0
69.9±0.8
71.7±0.6
76.9±0.9
77.8±0.7
78.2±0.5
Other
gpt-oss-20b
74.4±1.2
76.0±1.1
78.0±0.8
78.6±0.5
67.6±1.3
74.1±1.1
76.2±0.8
80.5±1.1
80.7±0.9
81.1±0.7
Qwen3-30B-A3B
68.7±1.4
71.2±1.2
73.8±0.8
74.2±0.5
62.5±1.4
69.6±1.1
72.4±0.7
75.9±1.2
78.6±0.8
79.7±0.6
DeepSeek-R1-8B
70.7±1.2
72.5±1.1
74.1±0.8
74.4±0.5
65.5±1.3
71.5±1.1
73.1±0.8
77.1±1.1
78.1±0.9
78.4±0.6
Average (models)
71.3±0.7
73.2±0.7
75.3±0.4
75.7±0.3
65.2±0.8
71.7±0.6
73.9±0.4
77.8±0.7
79.1±0.5
79.7±0.4
Average (tasks)
gpt-oss-20b
71.4±0.7
73.1±0.6
75.0±0.4
75.6±0.3
65.1±0.7
71.4±0.6
73.5±0.4
77.5±0.6
78.4±0.4
78.9±0.3
Qwen3-30B-A3B
65.3±0.7
67.6±0.6
70.0±0.4
70.2±0.3
61.0±0.6
66.7±0.5
68.8±0.4
72.4±0.6
74.9±0.5
75.8±0.3
DeepSeek-R1-8B
68.9±0.6
70.6±0.6
71.7±0.4
71.9±0.3
63.8±0.6
69.6±0.5
71.6±0.4
74.8±0.6
76.0±0.5
76.5±0.4
Average (models)
68.5±0.4
70.4±0.4
72.3±0.2
72.6±0.2
63.3±0.4
69.3±0.3
71.3±0.2
74.9±0.3
76.4±0.3
77.0±0.2
Table 6: AUROC of verbalized confidence (VC), self-consistency (SC), and their combination (SCVC) across
humanities tasks at different sampling budgets K. Rows report per-model results, with the “Average (models)” row
denoting the mean ± bootstrap standard deviation after averaging across models.
VC (K=1)
VC (K=2)
VC (K=5)
VC (K=8)
SC (K=2)
SC (K=5)
SC (K=8)
SCVC (K=2)
SCVC (K=5)
SCVC (K=8)
Math
gpt-oss-20b
94.1±0.3
94.0±0.3
94.5±0.2
94.5±0.1
94.0±0.3
94.5±0.2
94.5±0.1
94.0±0.3
94.5±0.2
94.5±0.1
Qwen3-30B-A3B
94.5±0.2
94.5±0.2
94.7±0.2
94.7±0.1
94.5±0.2
94.7±0.2
94.7±0.1
94.5±0.2
94.7±0.2
94.7±0.1
DeepSeek-R1-8B
91.8±0.4
91.8±0.4
92.8±0.2
92.9±0.2
91.8±0.4
92.8±0.2
92.9±0.2
91.8±0.4
92.8±0.2
92.9±0.2
Average (models)
93.5±0.2
93.4±0.2
94.0±0.1
94.0±0.1
93.4±0.2
94.0±0.1
94.0±0.1
93.4±0.2
94.0±0.1
94.0±0.1
GSM8K
gpt-oss-20b
94.2±0.3
94.1±0.3
94.8±0.2
94.8±0.1
94.1±0.3
94.8±0.2
94.8±0.1
94.1±0.3
94.8±0.2
94.8±0.1
Qwen3-30B-A3B
95.4±0.2
95.4±0.2
95.6±0.1
95.7±0.1
95.4±0.2
95.6±0.1
95.7±0.1
95.4±0.2
95.6±0.1
95.7±0.1
DeepSeek-R1-8B
92.9±0.3
92.9±0.3
93.7±0.2
93.8±0.1
92.9±0.3
93.7±0.2
93.8±0.1
92.9±0.3
93.7±0.2
93.8±0.1
Average (models)
94.1±0.2
94.1±0.2
94.7±0.1
94.8±0.1
94.1±0.2
94.7±0.1
94.8±0.1
94.1±0.2
94.7±0.1
94.8±0.1
AIME 2024 & 2025
gpt-oss-20b
95.8±2.0
95.8±1.9
98.2±0.4
98.2±0.1
95.8±1.9
98.2±0.4
98.2±0.1
95.8±1.9
98.2±0.4
98.2±0.1
Qwen3-30B-A3B
82.0±2.9
81.9±2.9
86.1±2.1
86.9±1.8
81.9±2.9
86.1±2.1
86.9±1.8
81.9±2.9
86.1±2.1
86.9±1.8
DeepSeek-R1-8B
66.1±3.7
66.0±3.8
73.7±2.5
75.2±2.1
66.0±3.8
73.7±2.5
75.2±2.1
66.0±3.8
73.7±2.5
75.2±2.1
Average (models)
81.3±1.6
81.2±1.7
86.0±1.1
86.8±0.9
81.2±1.7
86.0±1.1
86.8±0.9
81.2±1.7
86.0±1.1
86.8±0.9
Average (tasks)
gpt-oss-20b
94.7±0.7
94.7±0.7
95.8±0.1
95.8±0.1
94.7±0.7
95.8±0.1
95.8±0.1
94.7±0.7
95.8±0.1
95.8±0.1
Qwen3-30B-A3B
90.6±1.0
90.6±1.0
92.1±0.7
92.4±0.6
90.6±1.0
92.1±0.7
92.4±0.6
90.6±1.0
92.1±0.7
92.4±0.6
DeepSeek-R1-8B
83.6±1.3
83.6±1.3
86.8±0.8
87.3±0.7
83.6±1.3
86.8±0.8
87.3±0.7
83.6±1.3
86.8±0.8
87.3±0.7
Average (models)
89.6±0.6
89.6±0.6
91.6±0.4
91.9±0.3
89.6±0.6
91.6±0.4
91.9±0.3
89.6±0.6
91.6±0.4
91.9±0.3
Table 7: Accuracy of verbalized confidence (VC), self-consistency (SC), and their combination (SCVC) across
mathematics tasks at different sampling budgets K. Rows report per-model results, with the “Average (models)”
row denoting the mean ± bootstrap standard deviation after averaging across models.
12

## Page 13

VC (K=1)
VC (K=2)
VC (K=5)
VC (K=8)
SC (K=2)
SC (K=5)
SC (K=8)
SCVC (K=2)
SCVC (K=5)
SCVC (K=8)
Health
gpt-oss-20b
74.2±0.8
74.2±0.8
75.6±0.5
75.8±0.4
74.2±0.8
75.6±0.5
75.8±0.4
74.2±0.8
75.6±0.5
75.8±0.4
Qwen3-30B-A3B
75.4±0.6
75.4±0.6
76.4±0.4
76.5±0.3
75.4±0.6
76.4±0.4
76.5±0.3
75.4±0.6
76.4±0.4
76.5±0.3
DeepSeek-R1-8B
70.0±0.8
70.0±0.7
72.0±0.5
72.2±0.4
70.0±0.7
72.0±0.5
72.2±0.4
70.0±0.7
72.0±0.5
72.2±0.4
Average (models)
73.2±0.4
73.2±0.4
74.7±0.3
74.8±0.2
73.2±0.4
74.7±0.3
74.8±0.2
73.2±0.4
74.7±0.3
74.8±0.2
Biology
gpt-oss-20b
88.0±0.6
88.1±0.6
89.3±0.4
89.5±0.3
88.1±0.6
89.3±0.4
89.5±0.3
88.1±0.6
89.3±0.4
89.5±0.3
Qwen3-30B-A3B
88.7±0.5
88.7±0.5
89.4±0.2
89.4±0.2
88.7±0.5
89.4±0.2
89.4±0.2
88.7±0.5
89.4±0.2
89.4±0.2
DeepSeek-R1-8B
86.6±0.6
86.6±0.7
87.8±0.4
88.0±0.3
86.6±0.7
87.8±0.4
88.0±0.3
86.6±0.7
87.8±0.4
88.0±0.3
Average (models)
87.8±0.3
87.8±0.3
88.8±0.2
89.0±0.2
87.8±0.3
88.8±0.2
89.0±0.2
87.8±0.3
88.8±0.2
89.0±0.2
Chemistry
gpt-oss-20b
86.7±0.5
86.7±0.5
87.8±0.3
87.8±0.3
86.7±0.5
87.8±0.3
87.8±0.3
86.7±0.5
87.8±0.3
87.8±0.3
Qwen3-30B-A3B
88.2±0.4
88.2±0.4
88.4±0.2
88.4±0.2
88.2±0.4
88.4±0.2
88.4±0.2
88.2±0.4
88.4±0.2
88.4±0.2
DeepSeek-R1-8B
85.2±0.5
85.2±0.5
86.8±0.4
87.1±0.3
85.2±0.5
86.8±0.4
87.1±0.3
85.2±0.5
86.8±0.4
87.1±0.3
Average (models)
86.7±0.3
86.7±0.3
87.7±0.2
87.8±0.2
86.7±0.3
87.7±0.2
87.8±0.2
86.7±0.3
87.7±0.2
87.8±0.2
Economics
gpt-oss-20b
81.2±0.7
81.2±0.7
82.6±0.4
82.7±0.3
81.2±0.7
82.6±0.4
82.7±0.3
81.2±0.7
82.6±0.4
82.7±0.3
Qwen3-30B-A3B
84.4±0.5
84.4±0.5
84.7±0.3
84.8±0.2
84.4±0.5
84.7±0.3
84.8±0.2
84.4±0.5
84.7±0.3
84.8±0.2
DeepSeek-R1-8B
80.3±0.7
80.3±0.7
81.6±0.4
81.8±0.4
80.3±0.7
81.6±0.4
81.8±0.4
80.3±0.7
81.6±0.4
81.8±0.4
Average (models)
82.0±0.4
81.9±0.4
83.0±0.2
83.1±0.2
81.9±0.4
83.0±0.2
83.1±0.2
81.9±0.4
83.0±0.2
83.1±0.2
Physics
gpt-oss-20b
78.8±0.7
78.7±0.8
83.2±0.5
84.0±0.5
78.7±0.8
83.2±0.5
84.0±0.5
78.7±0.8
83.2±0.5
84.0±0.5
Qwen3-30B-A3B
89.6±0.4
89.6±0.4
90.1±0.2
90.2±0.2
89.6±0.4
90.1±0.2
90.2±0.2
89.6±0.4
90.1±0.2
90.2±0.2
DeepSeek-R1-8B
84.4±0.5
84.4±0.5
86.2±0.3
86.5±0.3
84.4±0.5
86.2±0.3
86.5±0.3
84.4±0.5
86.2±0.3
86.5±0.3
Average (models)
84.3±0.3
84.2±0.3
86.5±0.2
86.9±0.2
84.2±0.3
86.5±0.2
86.9±0.2
84.2±0.3
86.5±0.2
86.9±0.2
Computer Science
gpt-oss-20b
85.5±0.9
85.5±0.9
86.8±0.6
86.7±0.4
85.5±0.9
86.8±0.6
86.7±0.4
85.5±0.9
86.8±0.6
86.7±0.4
Qwen3-30B-A3B
85.7±0.7
85.8±0.7
86.3±0.4
86.3±0.3
85.8±0.7
86.3±0.4
86.3±0.3
85.8±0.7
86.3±0.4
86.3±0.3
DeepSeek-R1-8B
79.4±1.0
79.4±1.0
81.0±0.7
81.4±0.6
79.4±1.0
81.0±0.7
81.4±0.6
79.4±1.0
81.0±0.7
81.4±0.6
Average (models)
83.5±0.5
83.6±0.5
84.7±0.3
84.8±0.3
83.6±0.5
84.7±0.3
84.8±0.3
83.6±0.5
84.7±0.3
84.8±0.3
Engineering
gpt-oss-20b
68.9±0.8
68.9±0.8
71.4±0.5
71.8±0.4
68.9±0.8
71.4±0.5
71.8±0.4
68.9±0.8
71.4±0.5
71.8±0.4
Qwen3-30B-A3B
76.0±0.7
76.0±0.7
77.8±0.5
78.3±0.4
76.0±0.7
77.8±0.5
78.3±0.4
76.0±0.7
77.8±0.5
78.3±0.4
DeepSeek-R1-8B
67.6±0.9
67.6±0.9
71.7±0.6
72.3±0.4
67.6±0.9
71.7±0.6
72.3±0.4
67.6±0.9
71.7±0.6
72.3±0.4
Average (models)
70.9±0.4
70.8±0.4
73.6±0.3
74.1±0.2
70.8±0.4
73.6±0.3
74.1±0.2
70.8±0.4
73.6±0.3
74.1±0.2
GPQA Diamond
gpt-oss-20b
65.7±2.4
65.7±2.4
69.4±1.8
70.2±1.7
65.7±2.4
69.4±1.8
70.2±1.7
65.7±2.4
69.4±1.8
70.2±1.7
Qwen3-30B-A3B
69.8±1.9
69.7±1.8
70.7±1.4
70.8±1.3
69.7±1.8
70.7±1.4
70.8±1.3
69.7±1.8
70.7±1.4
70.8±1.3
DeepSeek-R1-8B
58.7±2.2
58.7±2.2
61.1±1.8
61.6±1.7
58.7±2.2
61.1±1.8
61.6±1.7
58.7±2.2
61.1±1.8
61.6±1.7
Average (models)
64.7±1.2
64.7±1.3
67.0±1.0
67.5±0.9
64.7±1.3
67.0±1.0
67.5±0.9
64.7±1.3
67.0±1.0
67.5±0.9
Average (tasks)
gpt-oss-20b
78.6±0.4
78.6±0.4
80.8±0.3
81.1±0.2
78.6±0.4
80.8±0.3
81.1±0.2
78.6±0.4
80.8±0.3
81.1±0.2
Qwen3-30B-A3B
82.2±0.3
82.2±0.3
83.0±0.2
83.1±0.2
82.2±0.3
83.0±0.2
83.1±0.2
82.2±0.3
83.0±0.2
83.1±0.2
DeepSeek-R1-8B
76.5±0.4
76.5±0.4
78.5±0.3
78.9±0.2
76.5±0.4
78.5±0.3
78.9±0.2
76.5±0.4
78.5±0.3
78.9±0.2
Average (models)
79.1±0.2
79.1±0.2
80.8±0.1
81.0±0.1
79.1±0.2
80.8±0.1
81.0±0.1
79.1±0.2
80.8±0.1
81.0±0.1
Table 8: Accuracy of verbalized confidence (VC), self-consistency (SC), and their combination (SCVC) across
STEM tasks at different sampling budgets K. Rows report per-model results, with the “Average (models)” row
denoting the mean ± bootstrap standard deviation after averaging across models.
13

## Page 14

VC (K=1)
VC (K=2)
VC (K=5)
VC (K=8)
SC (K=2)
SC (K=5)
SC (K=8)
SCVC (K=2)
SCVC (K=5)
SCVC (K=8)
Psychology
gpt-oss-20b
71.5±0.8
71.5±0.8
73.2±0.7
73.4±0.6
71.5±0.8
73.2±0.7
73.4±0.6
71.5±0.8
73.2±0.7
73.4±0.6
Qwen3-30B-A3B
76.9±0.6
76.8±0.6
77.5±0.5
77.5±0.4
76.8±0.6
77.5±0.5
77.5±0.4
76.8±0.6
77.5±0.5
77.5±0.4
DeepSeek-R1-8B
73.3±0.8
73.3±0.8
74.7±0.6
75.0±0.5
73.3±0.8
74.7±0.6
75.0±0.5
73.3±0.8
74.7±0.6
75.0±0.5
Average (models)
73.9±0.4
73.9±0.4
75.1±0.3
75.3±0.3
73.9±0.4
75.1±0.3
75.3±0.3
73.9±0.4
75.1±0.3
75.3±0.3
Law
gpt-oss-20b
44.0±0.9
44.0±0.9
46.4±0.6
46.9±0.5
44.0±0.9
46.4±0.6
46.9±0.5
44.0±0.9
46.4±0.6
46.9±0.5
Qwen3-30B-A3B
53.9±0.8
53.9±0.8
55.7±0.5
56.3±0.4
53.9±0.8
55.7±0.5
56.3±0.4
53.9±0.8
55.7±0.5
56.3±0.4
DeepSeek-R1-8B
43.3±0.9
43.3±0.9
45.3±0.6
45.5±0.5
43.3±0.9
45.3±0.6
45.5±0.5
43.3±0.9
45.3±0.6
45.5±0.5
Average (models)
47.1±0.5
47.1±0.5
49.2±0.3
49.6±0.3
47.1±0.5
49.2±0.3
49.6±0.3
47.1±0.5
49.2±0.3
49.6±0.3
Business
gpt-oss-20b
84.9±0.6
84.8±0.6
86.0±0.4
86.1±0.3
84.8±0.6
86.0±0.4
86.1±0.3
84.8±0.6
86.0±0.4
86.1±0.3
Qwen3-30B-A3B
84.8±0.5
84.8±0.5
85.3±0.3
85.3±0.2
84.8±0.5
85.3±0.3
85.3±0.2
84.8±0.5
85.3±0.3
85.3±0.2
DeepSeek-R1-8B
81.0±0.7
81.0±0.7
82.7±0.5
83.0±0.3
81.0±0.7
82.7±0.5
83.0±0.3
81.0±0.7
82.7±0.5
83.0±0.3
Average (models)
83.6±0.3
83.5±0.3
84.7±0.2
84.8±0.2
83.5±0.3
84.7±0.2
84.8±0.2
83.5±0.3
84.7±0.2
84.8±0.2
History
gpt-oss-20b
61.2±1.2
61.1±1.2
62.8±0.8
62.9±0.6
61.1±1.2
62.8±0.8
62.9±0.6
61.1±1.2
62.8±0.8
62.9±0.6
Qwen3-30B-A3B
65.2±0.9
65.2±0.9
66.2±0.6
66.3±0.4
65.2±0.9
66.2±0.6
66.3±0.4
65.2±0.9
66.2±0.6
66.3±0.4
DeepSeek-R1-8B
56.2±1.3
56.3±1.2
57.4±0.8
57.8±0.6
56.3±1.2
57.4±0.8
57.8±0.6
56.3±1.2
57.4±0.8
57.8±0.6
Average (models)
60.8±0.7
60.9±0.7
62.2±0.4
62.3±0.3
60.9±0.7
62.2±0.4
62.3±0.3
60.9±0.7
62.2±0.4
62.3±0.3
Philosophy
gpt-oss-20b
61.6±1.1
61.5±1.1
63.7±0.7
64.2±0.5
61.5±1.1
63.7±0.7
64.2±0.5
61.5±1.1
63.7±0.7
64.2±0.5
Qwen3-30B-A3B
67.8±0.9
67.7±0.9
68.9±0.6
69.3±0.5
67.7±0.9
68.9±0.6
69.3±0.5
67.7±0.9
68.9±0.6
69.3±0.5
DeepSeek-R1-8B
57.2±1.1
57.1±1.1
58.4±0.8
58.6±0.6
57.1±1.1
58.4±0.8
58.6±0.6
57.1±1.1
58.4±0.8
58.6±0.6
Average (models)
62.2±0.6
62.1±0.6
63.7±0.4
64.0±0.3
62.1±0.6
63.7±0.4
64.0±0.3
62.1±0.6
63.7±0.4
64.0±0.3
Other
gpt-oss-20b
66.8±0.8
66.7±0.8
69.1±0.6
69.4±0.5
66.7±0.8
69.1±0.6
69.4±0.5
66.7±0.8
69.1±0.6
69.4±0.5
Qwen3-30B-A3B
73.3±0.6
73.3±0.7
74.3±0.4
74.3±0.3
73.3±0.7
74.3±0.4
74.3±0.3
73.3±0.7
74.3±0.4
74.3±0.3
DeepSeek-R1-8B
64.0±0.8
64.0±0.8
66.0±0.5
66.4±0.4
64.0±0.8
66.0±0.5
66.4±0.4
64.0±0.8
66.0±0.5
66.4±0.4
Average (models)
68.0±0.4
68.0±0.5
69.8±0.3
70.0±0.2
68.0±0.5
69.8±0.3
70.0±0.2
68.0±0.5
69.8±0.3
70.0±0.2
Average (tasks)
gpt-oss-20b
65.0±0.4
65.0±0.4
66.9±0.3
67.2±0.2
65.0±0.4
66.9±0.3
67.2±0.2
65.0±0.4
66.9±0.3
67.2±0.2
Qwen3-30B-A3B
70.3±0.3
70.3±0.3
71.3±0.2
71.5±0.2
70.3±0.3
71.3±0.2
71.5±0.2
70.3±0.3
71.3±0.2
71.5±0.2
DeepSeek-R1-8B
62.5±0.4
62.5±0.4
64.1±0.3
64.4±0.2
62.5±0.4
64.1±0.3
64.4±0.2
62.5±0.4
64.1±0.3
64.4±0.2
Average (models)
65.9±0.2
65.9±0.2
67.4±0.1
67.7±0.1
65.9±0.2
67.4±0.1
67.7±0.1
65.9±0.2
67.4±0.1
67.7±0.1
Table 9: Accuracy of verbalized confidence (VC), self-consistency (SC), and their combination (SCVC) across
humanities tasks at different sampling budgets K. Rows report per-model results, with the “Average (models)” row
denoting the mean ± bootstrap standard deviation after averaging across models.
14

## Page 15

B
Appendix: Detailed prompts
 Give a confidence number from 1 
 to 100 that represents your overall 
 confidence that the **final   answer 
 is correct**.
(a) Vanilla uncertainty
 Give a confidence number from 1 to 100 that 
 represents how likely the **final answer is correct**, 
 *based on the quality of the reasoning*. Focus on 
 soundness, validity, and coherence. Pay attention to 
 errors, unsupported steps, flawed assumptions, etc.
    1   - Reasoning fatally flawed; **final answer is         
           almost certainly wrong**  
    25  - Major gaps or errors; **final answer is 
       
    probably wrong**  
    50  - Mixed support; **final answer is plausible but   
            not well-proved**  
    75  - Strong reasoning with only minor issues;     
 
            **final answer is probably right**  
    100 - Airtight reasoning; **final answer is correct**
 Give a confidence number from 1 to 100 that 
 represents **how confident you are in your final 
 answer**.
 Treat your final answer as fixed and do not try to re- 
 solve the problem. Focus primarily on your expressed 
 certainty or doubt—hedging, hesitation, self-correction.
    1   - You feel like you're making a random guess; 
 
           openly unsure  
    25  - You have significant doubts or hesitations  
    50  - Mixed feelings; some confidence, some      
 
            doubt  
    75  - Mostly confident with minor reservations  
    100 - Completely certain; no doubts whatsoever
(b) Verification uncertainty instructions
(c) Epistemic uncertainty instructions
Figure 4: Overview of uncertainty instructions prompts
defining VC methods. (a) Vanilla uncertainty instruc-
tion, (b) Verification uncertainty instruction and (c) epis-
temic uncertainty instruction. Each of the instructions is
used both for elicitation and judge methods. For judge
method, the epistemic uncertainty instructions are a bit
different, as it needs to pay attention to the solver’s rea-
soning trace, not its own.
This Section describes the detailed prompts, and
Figure 4 provides an overview of those used to
obtain answers and elicit confidence.
Next are given exact prompt descriptions.
Prompt 1 is used for getting model-elicited uncer-
tainties. Prompt 2 is used to get the LRM thought
trace without uncertainties, and after prompt 3
is used as a judge, giving us the Judge method.
Prompts 1, 2 and 3 are used for multiple-answer
questions. Prompts 8, 9 and 10 are like prompts 1,
2 and 3, but for the math dataset.
Inside the prompts 1, 2 and 3 (same for 8, 9 and
10), there are brackets which are used for inputting
variables. Question, choices, and letter correspond
to the question in hand from the dataset, but uncer-
tainty_instructions are subprompts which are based
on the method type: vanilla (prompt 4), verification
(prompt 5) and epistemic for the elicitation and the
judge method (prompt 6 and 7).
15

## Page 16

You are given a multiple choice question.
**Solve the problem**, showing your reasoning step
by step. After solving , provide your confidence
in your answer.
{uncertainty_instructions}
{question}
{choices}
Your response must *end* with exactly two lines of
the *exact* format below (no quotes) as the
very final lines of your answer:
'ANSWER: $LETTER '
'CONFIDENCE: $NUMBER '
- $LETTER must be one of the following options: {
letters}
Think step by step before answering and show your
reasoning first.
Listing 1: LRM prompt for multiple choice question
confidence elicitation
You are given a multiple choice question.
Solve the problem , showing your reasoning step by
step.
{question}
{choices}
Your response must *end* with exactly one line of
the *exact* format below (no quotes) as the
very final line of your answer:
'ANSWER: $LETTER '
- $LETTER must be one of the following options: {
letters}
Think step by step before answering and show your
reasoning first.
Listing 2: LRM prompt for multiple choice question
without uncertainty elicitation
You are given a problem along with a solver 's full
reasoning process and the answer they arrived
at.
{uncertainty_instructions}
[BEGIN PROBLEM]
{question}
[END PROBLEM]
[BEGIN REASONING AND ANSWER]
{reasoning_and_answer}
[END REASONING AND ANSWER]
Think step by step. End your final answer with this
exact format as the final line:
CONFIDENCE: [number between 1 and 100]
Listing 3: LRM prompt for multiple choice question for
judging solver’s reasoning process
Give a confidence number from 1 to 100 that
represents your overall confidence that the **
final answer is correct **.
Listing 4: Vanilla uncertainty prompt for uncertainty
instructions bracket in the main prompt
Give a confidence number from 1 to 100 that
represents how likely the ** final answer is
correct**, *based on the quality of the
reasoning *. Focus on soundness , validity , and
coherence. Pay attention to errors , unsupported
steps , flawed assumptions , etc.
1
- Reasoning fatally flawed; ** final answer
is almost certainly wrong **
25
- Major gaps or errors; ** final answer is
probably wrong **
50
- Mixed support; ** final answer is plausible
but not well -proved **
75
- Strong reasoning with only minor issues;
** final answer is probably right **
100 - Airtight reasoning; ** final answer is
correct **
Listing
5:
Verification
uncertainty
prompt
for
uncertainty instructions bracket in the main prompt
Give a confidence number from 1 to 100 that
represents **how confident you are in your
final answer **.
Treat your final answer as fixed and do not try to
re -solve the problem. Focus primarily on your
expressed certainty or doubt -hedging ,
hesitation , self -correction.
1
- You feel like you 're making a random guess
; openly unsure
25
- You have significant doubts or hesitations
50
- Mixed feelings; some confidence , some
doubt
75
- Mostly confident with minor reservations
100 - Completely certain; no doubts whatsoever
Listing 6: Epistemic uncertainty prompt for uncertainty
instructions bracket in the main prompt for elicitation
Give a confidence number from 1 to 100 that
represents **how confident the solver is in
their final answer **.
Treat their final answer as fixed and do not try to
re -solve the problem. Focus primarily on their
expressed certainty or doubt -hedging ,
hesitation , self -correction.
1
- They feel like they 're making a random
guess; openly unsure
25
- They have significant doubts or
hesitations
50
- Mixed feelings; some confidence , some
doubt
75
- Mostly confident with minor reservations
100 - Completely certain; no doubts whatsoever
Listing 7: Epistemic uncertainty prompt for uncertainty
instructions bracket in the main prompt for judge
16

## Page 17

You are given a math problem.
**Solve the problem**, showing your reasoning step
by step. After solving , provide your confidence
in your answer.
{uncertainty_instructions}
{prompt}
Your response must *end* with exactly two lines of
the *exact* format below (no quotes) as the
very final lines of your answer:
'ANSWER: $ANSWER '
'CONFIDENCE: $NUMBER '
- Do not use LaTeX boxes like \boxed in the final
lines; output plain text only.
- Think step by step before answering and show your
reasoning first.
Listing 8: LRM prompt for math question confidence
elicitation
You are given a math problem.
Solve the problem , showing your reasoning step by
step.
{prompt}
Your response must *end* with exactly one line of
the *exact* format below (no quotes) as the
very final line of your answer:
'ANSWER: $ANSWER '
- Do not use LaTeX boxes like \boxed in the final
line; output plain text only.
- Think step by step before answering and show your
reasoning first.
Listing 9: LRM prompt for math question without
uncertainty elicitation
You are given a problem along with a solver 's full
reasoning process and the answer they arrived
at.
{uncertainty_instructions}
[BEGIN PROBLEM]
{question}
[END PROBLEM]
[BEGIN REASONING AND ANSWER]
{reasoning_and_answer}
[END REASONING AND ANSWER]
Think step by step. End your final answer with this
exact format as the final line:
CONFIDENCE: [number between 1 and 100]
- Do not use LaTeX boxes like \boxed in the final
line; output plain text only.
Listing 10: LRM prompt for math question for judging
solver’s reasoning process
17
