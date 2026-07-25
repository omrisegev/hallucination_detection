---
source_pdf: papers/Uncertainty Under the Curve A Sequence-Level Entropy Area Metric for Reasoning LLM.pdf
slug: uncertainty-under-the-curve-a-sequence-level-entropy-area-me
pages: 8
extracted_on: 2026-07-13
---

# Uncertainty Under the Curve A Sequence-Level Entropy Area Metric for Reasoning LLM

## Page 1

Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for
Reasoning LLM
Yongfu Zhu, Lin Sun, Guangxiang Zhao, Weihong Lin, Xiangzheng Zhang
Qiyuan Tech
Abstract
In this work, we introduce Entropy Area Score (EAS), a sim-
ple yet effective metric to quantify uncertainty in the an-
swer generation process of reasoning large language models
(LLMs). EAS requires neither external models nor repeated
sampling, it integrates token-level predictive entropy from the
model itself to capture the evolution of uncertainty during
generation. Empirical results show that EAS is strongly cor-
related with answer entropy across models and datasets. In
training data selection, EAS identifies high-potential samples
and consistently outperforms Pass Rate filtering under equal
sample budgets, improving student model accuracy on math
benchmarks. EAS is both efficient and interpretable, offering
a practical tool for uncertainty modeling and data quality as-
sessment in LLM training.
1 Introduction
Reasoning LLMs have shown strong performance in com-
plex domains such as mathematics and science. However,
their outputs remain sensitive to minor changes in evalu-
ation conditions (e.g., random seeds, temperature, prompt
format), leading to significant fluctuations in reported scores
and undermining reproducibility (Hochlehnert et al. 2025;
Sun et al. 2025).
These fluctuations often stem from the model’s uncer-
tainty when solving ambiguous or borderline problems. Sta-
ble results occur only when a model consistently answers
correctly or incorrectly; in contrast, variance across runs re-
veals internal indecision. This highlights the need for re-
liable, fine-grained methods to quantify model uncertainty
during the reasoning process—both to improve evaluation
stability and to support downstream tasks such as training
data selection.
To address this, we propose Entropy Area Score (EAS),
a simple and efficient metric that tracks the evolution of un-
certainty throughout generation and provides actionable in-
sights into model behavior.
1.1 Related Work
Existing approaches for modeling and quantifying LLM un-
certainty can be broadly classified into two categories:
I.
Explicit
uncertainty
estimation
through
model-
internal or auxiliary mechanisms.
These methods typi-
cally involve training the model itself or an external scoring
model to estimate output uncertainty or confidence levels.
For example, Lin, Hilton, and Evans (2022) fine-tunes GPT-
3 to produce not only answers but also natural-language con-
fidence estimates (e.g., 90% confidence). Similarly, Kada-
vath et al. (2022) adds an auxiliary confidence head to the
model and trains it via confidence-based fine-tuning. Other
approaches, such as Tian et al. (2023), elicit confidence esti-
mates either directly or through a two-stage prompting pro-
cess.
Beyond model-internal confidence heads, some works in-
corporate external probing modules. For instance, Heo et al.
(2025) trains a linear classifier to map internal LLM repre-
sentations to task success, treating the classifier’s probabil-
ity as an uncertainty score. Liu et al. (2024) uses supervised
learning on labeled data, leveraging both hidden activations
and output entropy to predict the model’s confidence.
While effective in some settings, these methods rely heav-
ily on either model quality or external supervision. As noted
by Kapoor et al. (2024), lower-performing models tend to
produce overconfident yet incorrect predictions, limiting the
reliability of self-reported confidence. Moreover, training
task-specific uncertainty models is costly and lacks gener-
alizability across domains and model scales.
II. Implicit uncertainty estimation via statistical sig-
nals from model outputs.
This category avoids training
additional components and instead uses statistical proper-
ties—e.g., token-level log probabilities or entropy—to as-
sess uncertainty. A classical metric is perplexity (Jelinek
et al. 2005), proposed early on as a measure of linguistic
difficulty and prediction uncertainty. In machine translation,
Fomicheva et al. (2020) leverages output probabilities and
attention weights to calibrate confidence. Jiang et al. (2020)
focuses on multiple-choice QA and analyzes signals such as
the margin between top predicted options to evaluate answer
confidence.
Farquhar et al. (2024) samples multiple outputs and uses
entropy over semantic equivalence to quantify uncertainty.
These approaches have inspired our work by highlighting
how repeated outputs or token-level statistics can reflect
model behavior.
In reasoning-intensive benchmarks like mathematics or
science, answers are typically unique and unambiguous,
which makes these domains ideal for studying model uncer-
arXiv:2508.20384v1  [cs.AI]  28 Aug 2025

## Page 2

tainty. Moreover, we observe that powerful reasoning LLMs
like DeepSeek-R1-0528 frequently generate tokens such
as “Wait”, “But”, or “Alternatively” during math problem-
solving—indicative of a dynamic trial-and-error process.
For instance, in a random sample of 100K math examples
from AM-DeepSeek-R1-0528-Distilled (a-m team 2025),
the combined frequency of these three tokens is 0.98%, i.e.,
roughly once every 100 tokens.
Such patterns reveal that reasoning is not a static process
but one involving continual hypothesis revision. Yet, many
uncertainty metrics only consider output-final statistics (e.g.,
entropy of the last token), failing to capture the trajectory of
uncertainty as the model thinks and generates.
1.2 Our Contribution
To address these limitations, we propose a new metric: En-
tropy Area Score (EAS), which explicitly models the evo-
lution of token-level uncertainty across the generation path.
Our goal is to measure how confident the model is in its own
answer—not how correct the answer is which aligns with
the concept of distributional uncertainty (i.e., “not know-
ing what it doesn’t know”) discussed in (Malinin and Gales
2018).
Although it does not directly measure answer correct-
ness, such uncertainty quantification remains valuable—for
instance, in training data selection. A popular strategy, seen
in works like Lyu et al. (2025) uses Pass Rate filtering:
only retaining examples with intermediate accuracy (neither
all-correct nor all-wrong) during multi-sample reasoning.
This approach has proven effective across multiple reason-
ing LLMs training (Team et al. 2025; DeepSeek-AI 2025;
Wen et al. 2025).
However, it suffers from logical limitations and high com-
putational cost due to repeated sampling. In contrast, our
proposed EAS only requires a single forward pass and con-
sistently outperforms baseline strategies in both accuracy
and efficiency across architectures and model sizes.
In summary, our key contributions are as follows:
1. EAS: A novel metric for modeling uncertainty in lan-
guage model outputs. EAS is simple to compute, re-
quiring no auxiliary models or fine-tuning. It directly
leverages the model’s native token-level predictions, en-
abling generalization across tasks and models with min-
imal cost. Moreover, it provides a dynamic trajectory of
uncertainty, offering fine-grained interpretability.
2. Demonstrated correlation with sampling-based un-
certainty. We show that EAS strongly correlates with
answer entropy derived from repeated generation across
multiple models and tasks, validating it as a reliable
proxy for output uncertainty.
3. Effective application in data selection for training. By
identifying samples where the model exhibits high un-
certainty during generation, EAS helps select data with
high learning potential. Compared with random, length-
based, or Pass Rate-based selection, EAS consistently
improves student model performance under the same
sample size, showing its practical utility in large-scale
training pipelines.
2 Entropy Area Score (EAS)
Figure 1: Token-Level Entropy Trajectory and EAS
Computation. (1) The shaded area under the entropy curve
represents the EAS score, reflecting cumulative uncertainty.
(2) In the early stages, the model shows no clear preference,
leading to higher entropy. As generation progresses and the
model converges on a likely answer, entropy decreases.
To quantify uncertainty in the answer generation process of
reasoning models, especially in domains like mathematics
and science, we formally define the EAS as follows:
• Step 1: Context Construction. Let the model’s gener-
ated token sequence be S = (x1, x2, . . . , xT , . . . ), where
xT is the last token of the final answer (note that it may
not be the last token of the full sequence). At each posi-
tion t ∈{1, 2, . . . , T −1}, we construct a context ˜Ct for
predicting the next token as:
˜Ct = (x1, x2, . . . , xt, “\boxed{”, prefix(L−1)
ans
)
• Step 2: Entropy Computation. At each position t, the
model produces a predictive distribution over the vocabu-
lary V: Pt(v) = P(v | ˜Ct),
v ∈V. The corresponding
token-level entropy is then defined as:
Ht = −
X
v∈V
Pt(v) log2 Pt(v)
• Step 3: Area Integral of Entropy (EAS). We compute
the sum of entropy values from position 1 to T −1 to

## Page 3

capture the total uncertainty across the generation trajec-
tory:
EAS(S) =
T −1
X
t=1
Ht
3 Comparison with Other Metrics
To assess the effectiveness of EAS, we compare it against
other commonly used and lightweight uncertainty metrics.
3.1 Experimental Setup
We run 64 inferences with DeepSeek-R1-Distill-Qwen-14B
on the GPQA-Diamond dataset (198 science questions) to
compute sampling-based uncertainty. A standard metric is
correctness entropy, which treats each output as correct or
incorrect and measures uncertainty as:
Hcorrectness = −p log2 p −(1 −p) log2(1 −p)
where p = nc
N is the proportion of correct answers among
the N = 64 samples.
However, this binary view lacks granularity—it cannot
distinguish between consistently wrong predictions and var-
ied incorrect answers. Since we aim to capture the model’s
internal uncertainty, we instead use answer entropy, which
reflects the diversity of generated answers:
Let A = {a1, a2, . . . , aK} denote the set of unique an-
swers generated by the model across the 64 runs, and nk be
the count of answer ak. The answer entropy is then com-
puted as:
Hanswer = −
K
X
k=1
pk log2 pk,
pk = nk
N
This reference metric reflects how diverse or unstable the
model’s outputs are under repeated inference. Low answer
entropy indicates consistent outputs; high entropy reveals in-
decision or instability.
To evaluate different uncertainty metrics, we compute
each metric per question and report its Pearson correlation
with answer entropy across all 198 samples. This correlation
quantifies how well each metric approximates sampling-
based uncertainty. We compare the following metrics:
• EAS: Computed via a single forward pass, EAS inte-
grates token-level entropy across the generation path.
At each step, we construct a context suffix including
“\boxed{” and the first L −1 tokens of the ground-truth
answer, and use the vLLM (Kwon et al. 2023) API to
extract top-K token probabilities. To account for trunca-
tion, we estimate the maximum entropy error where V =
151,665 and K = 20 as:
∆Hmax ≤ε · log2
V −K
ε

,
ε = 1 −
K
X
i=1
pi
As shown in Table 1, top-20 tokens capture over 99.87%
of probability mass on average, limiting entropy error
to below 0.031 bits. Given average entropy is approx-
imately 0.66, this approximation introduces negligible
Mertic
Total Probability
Theoretical
Mass of Top-K
Error Bound
All Tokens Average
99.87%
0.031
90th Percentile Average
99.85%
0.040
95th Percentile Average
99.52%
0.12
99th Percentile Average
97.54%
0.56
Table 1: Upper Bound of Entropy Truncation Error in
EAS. Estimated maximum error from truncating the token
distribution to top-K (K=20), based on the principle of max-
imum entropy. Results show minimal distortion, validating
the approximation used in EAS.
distortion less than 4.70%. These findings confirm that
using a truncated probability distribution for EAS offers
a sound and robust approximation while greatly improv-
ing computational efficiency.
• Mean Entropy Area Score (Mean EAS): This is simply
the average entropy across token positions:
Mean EAS(S) =
1
T −1
T −1
X
t=1
 
−
X
v∈O
Pt(v) log2 Pt(v)
!
• Perplexity (PPL): This metric quantifies the model’s
perplexity over its own generated token sequence, reflect-
ing the overall uncertainty in its predictions:
PPL(S) = exp

−1
|S|
|S|
X
t=1
log P(xt | x<t)


• Response Length: The total number of tokens |S| in the
model’s response sequence.
3.2 Experimental Results and Analysis
We visualize the correlation between answer entropy and
four uncertainty metrics—EAS, Mean EAS, PPL, and Re-
sponse Length—in Figure 2. Among them, EAS shows the
strongest linear correlation, suggesting it best reflects model
uncertainty under repeated inference. This is because an-
swer entropy captures the consistency of model preferences
across runs, while EAS integrates uncertainty across the en-
tire generation path.
To enhance prediction salience and reduce noise from
low-probability tokens, we append the special suffix
“\boxed{” to guide answer generation. To further interpret
EAS’s behavior, we introduce a visualization based on de-
cayed cumulative option probabilities, which reveals how
the model’s preference over A/B/C/D evolves during gener-
ation. This method highlights the model’s internal decision
dynamics in multiple-choice settings.
We begin by formally defining the decayed cumulative
probability formulation that enhances the interpretability of
model preference trajectories.
1. At each generation step t, let the model’s predicted prob-
ability distribution over the multiple-choice options be

## Page 4

Figure 2: Correlation Between Uncertainty Metrics and Answer Entropy. Each point represents a GPQA-Diamond ques-
tion. Metrics are Z-score normalized. The red line indicates linear regression; Pearson’s r and p-value are shown per subplot.
EAS shows the strongest correlation, indicating it best captures model uncertainty.
P (O)
t
= {Pt(A), Pt(B), Pt(C), Pt(D)}. This gives us a
snapshot of the model’s preference over the answer op-
tions at time t, based on the context prefix ˜Ct defined ear-
lier. However, directly plotting P (O)
t
across time results
in highly oscillatory curves due to sensitivity to the im-
mediate context. These fluctuations obscure the true evo-
lution of preference, as temporary alignment between the
context and a specific option token can artificially spike
its predicted probability.
2. To smooth the noise and highlight overall trends, we
compute the cumulative sum of predicted probabilities
for each option up to step t:
bP (O)
t
(v) =
t
X
k=1
Pk(v),
∀v ∈{A, B, C, D}
This formulation suppresses short-term volatility and
emphasizes which option the model consistently leans
toward. However, it treats all positions equally and may
mask the influence of recent evidence—especially when
the model shifts its preference mid-generation. Further-
more, since each token’s value becomes a first-order dif-
ference of the cumulative curve, it is harder to observe
relative probability changes among competing options.
3. To capture both the smoothness and time sensitivity of
model preferences, we define a decayed cumulative prob-
ability with distance-based weighting:
ePt(O)(v) =
t
X
k=1
1
(t −k + 1)α · Pk(v)
Here, α > 0 is a decay coefficient that controls how
much weight is assigned to earlier positions. When α =
0, this reduces to the unweighted cumulative probabil-
ity (all positions equally weighted); when α →+∞,
it reduces to the raw one-step prediction P (O)
t
, focus-
ing entirely on the current token. By setting α = 0.5,
we found an effective trade-off between smoothing and
local responsiveness. This value ensures that the final an-
swer’s curve is typically dominant at the last token (i.e.,
the model’s prediction is visually validated), while pre-
serving readable trends over time.
We then discretize the questions based on their answer
entropy using a decay coefficient of 0.5, and examine intra-

## Page 5

Figure 3: Model Behavior Across Different Levels of Answer Entropy. Questions are grouped by answer entropy: low
[0, 0.5), medium [0.5, 1.5], and high (1.5, +∞), with sample counts 60:97:51. (1) Low-entropy examples show early and
stable preference for the correct option. (2) High-entropy cases exhibit frequent option switches and fluctuating uncertainty. (3)
Medium-entropy examples display partial stability followed by revision. Token-level entropy curves confirm that earlier and
wider uncertainty leads to higher EAS scores. These trends explain the strong alignment between EAS and answer entropy.
bucket similarities and inter-bucket differences. As illus-
trated in the upper portion of Figure 3, we observe several
consistent behavioral patterns across entropy levels:
• Observation 1: The model exhibits a strong and stable
preference for the final answer option in low-entropy
samples. In the [0, 0.5) group, the model quickly com-
mits to the final answer, with its probability curve rising
early and remaining dominant. Competing options show
only brief, minor fluctuations. This reflects strong inter-
nal confidence and results in consistent outputs across
runs. However, such confidence may be misplaced: 4 of
the 60 questions in this group scored 0, indicating over-
confident hallucination.
• Observation 2: No option maintains a consistent lead
in high-entropy examples. In the (1.5, +∞) group, the
model’s preference shifts frequently, with option curves
intersecting throughout the sequence. This indicates in-
decision and sensitivity to small perturbations. The re-
sulting outputs vary widely across runs, and only 4.4%
of these questions exceeded the average Pass@1 score of
57.64%.
• Observation 3: Medium-entropy samples displays
mixed dynamics. In the [0.5, 1.5] group, one option of-
ten shows an early lead, but is later overtaken by another
option. The model may initially favor a plausible answer,
then revise its judgment after generating additional con-
text or reaching a key reasoning step. Compared to low-
entropy samples, the turning point where the dominant
curve emerges typically occurs later. Compared to high-
entropy samples, the number of curve intersections is
fewer, and full tie scenarios (where no curve dominates)
are rare. These samples reflect partial stability with lo-
cal revision, where the model starts confidently but later
shifts its hypothesis—a behavior that aligns with mid-
range answer entropy.
From the above case studies, we can conclude that a lower
answer entropy is typically observed when the leading op-
tion curve begins rising earlier in the generation process, ex-
hibits denser upward segments, and has fewer intersections
with other option curves.
As shown in the lower part of Figure 3, these char-
acteristics are directly reflected in the token-level entropy
curves: low-entropy examples tend to have longer segments
where entropy remains low, while high-entropy examples
display sharp increases earlier in the sequence and across
a broader range. Consequently, the total accumulated uncer-
tainty—quantified by the EAS—is smaller for low-entropy
samples and larger for high-entropy samples, reinforcing the
observed positive correlation between EAS and answer en-
tropy.
To understand why EAS outperforms other uncertainty
metrics in capturing this behavior, we compare it against the
remaining baselines:
• Comparison to Mean EAS: Mean EAS, which sim-
ply averages the token-level entropy across the sequence,
fails to distinguish medium-entropy cases where the
model initially shows strong preference for one option
but later reverses course. In such cases, the entropy at
each position may remain low because the model main-
tains confidence at each step—even though its final an-
swer is unstable across runs. This leads to Mean EAS
underestimating uncertainty, and in some cases assigning
lower scores to medium-entropy examples than to truly
stable (low-entropy) ones. In contrast, EAS, as an inte-

## Page 6

gral over the entire entropy curve, captures not just the
momentary uncertainty but also the extent and duration
of uncertainty over time, effectively distinguishing these
cases.
• Comparison to PPL: PPL measures the model’s uncer-
tainty in language generation at the token level. A low
PPL indicates high fluency or grammatical consistency,
but not necessarily high confidence in the semantic cor-
rectness of the answer. In other words, a model may gen-
erate a smooth, well-formed response while still being
uncertain about the answer. Hence, PPL and answer en-
tropy are not causally linked, and the correlation between
PPL and uncertainty is weak in our experiments.
• Comparison to Response Length: Response length is
sometimes used as a proxy for uncertainty, assuming
longer outputs indicate more reasoning. However, this
correlation is inconsistent and task-dependent. In GPQA-
Diamond, one organic chemistry question led to out-
puts exceeding 6,300 tokens—longer than 6% of sam-
ples—yet achieved 100% Pass@1 with zero answer en-
tropy. This length was due to long chemical formulas
(e.g., C6H12O2 being tokenized into 7 subwords). In
such cases, response length reflects input structure rather
than uncertainty. Overall, its correlation with answer en-
tropy is weaker and noisier than that of EAS.
EAS consistently shows the strongest correlation with
sampling-based answer entropy, as it captures both local
hesitation and global uncertainty across the generation pro-
cess. This makes it a reliable and interpretable proxy for
model uncertainty in complex reasoning tasks.
3.3 Generalization Across Models and Tasks
In addition to the experiments above, we further evaluate the
generality and robustness of the EAS metric under varying
model scales, architectures, and task types. It is worth noting
that all selected models and tasks fall within the domain of
reasoning, which aligns with the objective of this study.
The evaluated models include various sizes and architec-
tures:
• Model families: Qwen2.5 and LLaMA
• Parameter scales: Ranging from 8B to 14B
The selected tasks are:
• AIME24 and AIME25: Two mathematics competitions
benchmarks, which we merge and report as a combined
set
• GPQA-Diamond: A science-based QA dataset
The results are summarized in Table 2. These results con-
firm that EAS maintains strong and significant correlation
with answer entropy across different architectures, param-
eter sizes, and reasoning tasks. This suggests that EAS is
a stable and generalizable uncertainty metric for evaluating
large language models on complex reasoning problems.
4 Training Data Selection Based on EAS
Having established EAS as a reliable uncertainty metric, we
explore its utility in training data selection—a critical chal-
lenge in LLM training, where noisy or low-quality samples
Model
Average AIME
GPQA-Diamond
DS-R1-Qwen-14B
0.8237
0.5968
DS-R1-LLaMA-8B
0.6820
0.5434
Table 2: Correlation Between EAS and Answer Entropy
Across Models and Tasks. Pearson correlation coefficients
for EAS vs. answer entropy on AIME and GPQA-Diamond
benchmarks using different DeepSeek-R1-Distill models.
All correlations are statistically significant (p < 5e −5).
Results show that EAS maintains strong correlation across
model scales and architectures.
can waste resources and hinder performance. By modeling
uncertainty across the generation trajectory, EAS helps dis-
tinguish between easy, hard, and ambiguous examples, mak-
ing it suitable for filtering and curriculum design. We vali-
date this through an ablation study under controlled settings.
4.1 Experimental Setup
We evaluate the effectiveness of EAS-based data selection
in a supervised fine-tuning (SFT) setup, controlling data
selection as the only variable. Experiments are conducted
on two base models: DeepSeek-R1-Distill-Qwen-14B (DS-
R1-Qwen-14B) and DeepSeek-R1-Distill-LLaMA-8B (DS-
R1-LLaMA-8B), using the math subset of AM-DeepSeek-
R1-0528-Distilled, which has already undergone correctness
verification, de-duplication, and contamination filtering. We
further apply vector-based contamination filtering and re-
move samples exceeding 20,480 tokens.
In the data selection strategies, we compared:
• Random Sampling: Uniformly sample 5,000 examples
as a baseline and ensures equal sample size across strate-
gies to eliminate size-based bias.
• Length-Based Selection: Sort samples in descending
order by total token length, motivated by the observa-
tion (e.g., in OpenAI-o1, DeepSeek-R1) that longer sam-
ples often include more complex context, semantic struc-
tures, or reasoning chains, potentially contributing more
to model training.
• Pass Rate-Based Selection: For each training sample,
we conduct 4 rounds of repeated inference and compute
Pass Rate as the ratio of correct outputs. Then we filter
out samples with Pass Rate = 1 (too easy) or 0 (too hard).
Among the remaining samples, retain those with lowest
Pass Rates, which are hypothesized to be challenging but
learnable.
• EAS-Based Selection: For each training sample, run one
forward pass using the base model and compute its EAS.
Then we retain the top 5,000 samples with the highest
EAS scores and these are considered the samples where
the model showed the most internal uncertainty during
generation, indicating potential learning value.
All selected samples are trained using identical hyperpa-
rameters under the ms-swift framework (Zhao et al. 2024).
For evaluation, we use the evalscope (Team 2024) frame-
work to measure performance on the AIME24 and AIME25

## Page 7

Model
Checkpoint-Rank
Random Sampling
Length-Based
Pass Rate-Based
EAS-Based
DS-R1-LLaMA-8B
1
56.8
+1.4
-2.5
+2.1
2
56.6
+1.5
-3.3
+2.3
3
56.3
+1.7
-3.3
+1.5
DS-R1-Qwen-14B
1
77.0
+0.3
+0.7
+1.2
2
76.7
+0.4
+0.8
+1.2
3
76.4
+0.7
+0
+1.3
Table 3: Performance Comparison of Different Data Selection Strategies. Average Pass@1 scores on AIME24 and AIME25
using four data selection strategies. Each model reports results from its top 3 validation checkpoints. EAS-based selection
consistently achieves the best performance, highlighting its effectiveness in identifying high-potential training samples via
single-pass uncertainty estimation.
datasets, each with 32 rounds of repeated inference. The
metric reported is Average Pass@1, using the same infer-
ence parameters as the original base models.
4.2 Experimental Results and Analysis
The results in Table 3 demonstrate that EAS-based data se-
lection consistently achieves the best performance across all
checkpoints and outperforms both Pass Rate and Length-
based strategies.
We attribute EAS’s advantage over Pass Rate to the fol-
lowing key factors:
I. EAS provides a finer-grained, single-pass uncertainty
estimate.
Pass Rate depends on multiple rounds of re-
peated inference and is constrained by discrete values (e.g.,
0, 0.25, 0.5, 0.75, 1 when using 4 samples). This coarse gran-
ularity makes it hard to distinguish between subtly different
samples. For instance, a sample where the model wavers be-
tween correct and incorrect options may still be categorized
the same as a highly confident one.
Moreover, the common practice of filtering out Pass Rate
= 1 and 0 samples and selecting those with low but non-zero
Pass Rates introduces logical inconsistencies. In our earlier
evaluation, among the 198 GPQA-Diamond questions, 44
had all 4 initial responses incorrect—yet 9 of them produced
at least 1 correct answer in the next 4 runs. These cases,
under Pass Rate = 4 setting, would be wrongly discarded as
“unsolvable,” despite having actual learning value.
By contrast, EAS is computed from a single forward pass,
is continuous-valued, and captures token-level uncertainty
across the entire output trajectory. It distinguishes between
consistently wrong-but-confident samples and those with
meaningful internal struggle—i.e., the “high-potential” sam-
ples.
II. EAS focuses on model–data interaction, not just
model correctness.
Pass Rate only measures whether the
model gets the right answer during repeated inference. It re-
flects outcome, but not the internal process or pedagogical
value of the sample.
EAS, in contrast, reflects how much uncertainty the model
experiences while generating the answer. It answers: “Did
this example make the model think?” rather than “Did the
model get it right?”
This allows EAS to identify training examples that stim-
ulate discriminative reasoning, contain ambiguity, or trigger
hypothesis revision—qualities that are valuable for learning.
In this sense, EAS can be interpreted as a model-data com-
patibility score, which helps prioritize samples that are nei-
ther too easy nor too hard, but rich in learning signals.
5 Discussion
In this paper, we proposed Entropy Area Score (EAS) as a
metric to quantify the uncertainty exhibited by large reason-
ing models during answer generation. Compared to existing
approaches, EAS offers both computational efficiency and
interpretability. We demonstrated its ability to capture shifts
in model preference and to identify uncertain samples across
complex reasoning tasks in mathematics and science.
However, EAS is inherently designed to measure the dis-
tributional uncertainty over answer tokens by integrating
token-level entropy during generation. This design makes it
less applicable to tasks with non-unique or structure-diverse
outputs, such as IFEval or LiveCodeBench, where the an-
swer is typically a free-form text or code segment. In these
tasks, correctness is judged based on semantic equivalence
or functional consistency, rather than the probability of gen-
erating a specific token or option.
As a result, the local uncertainty signals captured by EAS
may fail to reflect the true global uncertainty of the model
in such settings, leading to a lower correlation between EAS
and empirical outcome variability.
Nonetheless,
we
believe
EAS
presents
a
novel,
lightweight, and interpretable tool for uncertainty modeling
in LLMs. It performs well not only in evaluation scenarios,
but also proves useful in practical applications such as
training data selection and curriculum optimization. In
the future, we aim to extend EAS toward a more general
uncertainty modeling framework applicable to broader
generation tasks beyond discrete option prediction.
References
a-m team. 2025. AM-DeepSeek-R1-0528-Distilled.
DeepSeek-AI. 2025.
DeepSeek-R1: Incentivizing Rea-
soning Capability in LLMs via Reinforcement Learning.
arXiv:2501.12948.

## Page 8

Farquhar, S.; Kossen, J.; Kuhn, L.; and Gal, Y. 2024. Detect-
ing hallucinations in large language models using semantic
entropy. Nature, 630(8017): 625–630.
Fomicheva, M.; Sun, S.; Yankovskaya, L.; Blain, F.;
Guzm´an, F.; Fishel, M.; Aletras, N.; Chaudhary, V.; and Spe-
cia, L. 2020. Unsupervised Quality Estimation for Neural
Machine Translation. Transactions of the Association for
Computational Linguistics, 8: 539–555.
Heo, J.; Xiong, M.; Heinze-Deml, C.; and Narain, J.
2025. Do LLMs estimate uncertainty well in instruction-
following? arXiv:2410.14582.
Hochlehnert, A.; Bhatnagar, H.; Udandarao, V.; Albanie, S.;
Prabhu, A.; and Bethge, M. 2025. A Sober Look at Progress
in Language Model Reasoning: Pitfalls and Paths to Repro-
ducibility. arXiv:2504.07086.
Jelinek, F.; Mercer, R. L.; Bahl, L. R.; and Baker, J. K. 2005.
Perplexity—a measure of the difficulty of speech recogni-
tion tasks. The Journal of the Acoustical Society of America,
62(S1): S63–S63.
Jiang, Z.; Xu, F. F.; Araki, J.; and Neubig, G. 2020. How Can
We Know What Language Models Know? Transactions of
the Association for Computational Linguistics, 8: 423–438.
Kadavath, S.; Conerly, T.; Askell, A.; Henighan, T.; Drain,
D.; Perez, E.; Schiefer, N.; Hatfield-Dodds, Z.; DasSarma,
N.; Tran-Johnson, E.; Johnston, S.; El-Showk, S.; Jones, A.;
Elhage, N.; Hume, T.; Chen, A.; Bai, Y.; Bowman, S.; Fort,
S.; Ganguli, D.; Hernandez, D.; Jacobson, J.; Kernion, J.;
Kravec, S.; Lovitt, L.; Ndousse, K.; Olsson, C.; Ringer, S.;
Amodei, D.; Brown, T.; Clark, J.; Joseph, N.; Mann, B.; Mc-
Candlish, S.; Olah, C.; and Kaplan, J. 2022. Language Mod-
els (Mostly) Know What They Know. arXiv:2207.05221.
Kapoor, S.; Gruver, N.; Roberts, M.; Collins, K.; Pal, A.;
Bhatt, U.; Weller, A.; Dooley, S.; Goldblum, M.; and Wil-
son, A. G. 2024. Large Language Models Must Be Taught to
Know What They Don’t Know. In Globerson, A.; Mackey,
L.; Belgrave, D.; Fan, A.; Paquet, U.; Tomczak, J.; and
Zhang, C., eds., Advances in Neural Information Process-
ing Systems, volume 37, 85932–85972. Curran Associates,
Inc.
Kwon, W.; Li, Z.; Zhuang, S.; Sheng, Y.; Zheng, L.; Yu,
C. H.; Gonzalez, J. E.; Zhang, H.; and Stoica, I. 2023. Ef-
ficient Memory Management for Large Language Model
Serving with PagedAttention. In Proceedings of the ACM
SIGOPS 29th Symposium on Operating Systems Principles.
Lin, S.; Hilton, J.; and Evans, O. 2022. Teaching Models to
Express Their Uncertainty in Words. arXiv:2205.14334.
Liu, L.; Pan, Y.; Li, X.; and Chen, G. 2024. Uncertainty Es-
timation and Quantification for LLMs: A Simple Supervised
Approach. arXiv:2404.15993.
Lyu, C.; Gao, S.; Gu, Y.; Zhang, W.; Gao, J.; Liu, K.; Wang,
Z.; Li, S.; Zhao, Q.; Huang, H.; Cao, W.; Liu, J.; Liu, H.;
Liu, J.; Zhang, S.; Lin, D.; and Chen, K. 2025. Exploring
the Limit of Outcome Reward for Learning Mathematical
Reasoning. arXiv:2502.06781.
Malinin, A.; and Gales, M. 2018. Predictive Uncertainty Es-
timation via Prior Networks. In Bengio, S.; Wallach, H.;
Larochelle, H.; Grauman, K.; Cesa-Bianchi, N.; and Gar-
nett, R., eds., Advances in Neural Information Processing
Systems, volume 31. Curran Associates, Inc.
Sun, L.; Lin, W.; Wu, J.; Zhu, Y.; Jian, X.; Zhao, G.; Jia,
C.; Zhang, L.; er Hu, S.; Wu, Y.; and Zhang, X. 2025.
Evaluation is All You Need: Strategic Overclaiming of
LLM Reasoning Capabilities Through Evaluation Design.
arXiv:2506.04734.
Team, K.; Bai, Y.; Bao, Y.; and et al. 2025. Kimi K2: Open
Agentic Intelligence. arXiv:2507.20534.
Team, M. 2024.
EvalScope: Evaluation Framework for
Large Models.
Tian, K.; Mitchell, E.; Zhou, A.; Sharma, A.; Rafailov, R.;
Yao, H.; Finn, C.; and Manning, C. 2023.
Just Ask for
Calibration: Strategies for Eliciting Calibrated Confidence
Scores from Language Models Fine-Tuned with Human
Feedback. In Bouamor, H.; Pino, J.; and Bali, K., eds., Pro-
ceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, 5433–5442. Singapore: As-
sociation for Computational Linguistics.
Wen, L.; Cai, Y.; Xiao, F.; He, X.; An, Q.; Duan, Z.; Du,
Y.; Liu, J.; Tang, L.; Lv, X.; Zou, H.; Deng, Y.; Jia, S.; and
Zhang, X. 2025. Light-R1: Curriculum SFT, DPO and RL
for Long COT from Scratch and Beyond. arXiv:2503.10460.
Zhao, Y.; Huang, J.; Hu, J.; Wang, X.; Mao, Y.; Zhang, D.;
Jiang, Z.; Wu, Z.; Ai, B.; Wang, A.; Zhou, W.; and Chen,
Y. 2024. SWIFT:A Scalable lightWeight Infrastructure for
Fine-Tuning. arXiv:2408.05517.
