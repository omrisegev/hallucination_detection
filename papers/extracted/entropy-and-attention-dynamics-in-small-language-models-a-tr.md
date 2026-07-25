---
source_pdf: papers/Entropy and Attention Dynamics in Small Language Models A Trace-Level Structural Analysis on the TruthfulQA Benchmark.pdf
slug: entropy-and-attention-dynamics-in-small-language-models-a-tr
pages: 24
extracted_on: 2026-07-13
---

# Entropy and Attention Dynamics in Small Language Models A Trace-Level Structural Analysis on the TruthfulQA Benchmark

## Page 1

Entropy and Attention Dynamics in Small
Language Models: A Trace-Level Structural
Analysis on the TruthfulQA Benchmark
Adeyemi Adeseye1, Aisvarya Adeseye2, Hannu Tenhunen3, and Jouni Isoaho4
1 Brilloconnetz Partners avoin yhti¨o, Turku, Finland
adeyemi@brilloconnetz.com
2 University of Turku, Turku, Finland
aisvarya.a.adeseye@utu.fi
3 Royal Institute of Technology, Stockholm, Sweden
hannu@kth.se
4 University of Turku, Turku, Finland
jouni.isoaho@utu.fi
Abstract. —
Small language models (SLMs) have been increasingly deployed in edge
devices and other resource-constrained settings. However, these mod-
els make confident mispredictions and produce unstable output, mak-
ing them risky for factual and decision-critical tasks. Current evaluation
methodology relies on final accuracy or hallucination rates without ex-
plaining how internal model behavior affects outputs. Specifically, how
entropy evolves during decoding, how attention is distributed across lay-
ers, and how hidden representations contribute to uncertainty, logical
inconsistencies, and misinformation propagation are often overlooked.
Consequently, this study introduces a trace-level analysis of entropy and
attention dynamics in SLMs evaluated with the TruthfulQA dataset.
Four models with parameter ranges of 1B-1.7B parameters were exam-
ined via token-level output entropy, attention entropy, head dispersion,
and hidden-state representation. The results reflect three model clas-
sifications by entropy patterns. Deterministic models (DeepSeek-1.5B
and LLaMA-1B): output entropy decreases over time. Exploratory mod-
els (Gemma-1B): with increasing entropy, and balanced models (Qwen-
1.7B): have moderate and stable entropy. Also, each group has distinc-
tively different hidden-state movement and attention dispersion patterns.
The analysis demonstrates that truthfulness in SLMs emerges from struc-
tured entropy and attention dynamics. Monitoring and optimizing these
internal uncertainty patterns can guide the design of a more reliable,
hallucination-aware, and application-specific edge SLMs.
Keywords: Small Language Models (SLMs), Entropy Dynamics, At-
tention Mechanisms, Truthfulness Evaluation, Uncertainty Analysis
arXiv:2604.03589v1  [cs.AI]  4 Apr 2026

## Page 2

2
A. Adeseye et al
1
Introduction
Small language models (SLMs) are now often utilized in edge devices, privacy-
sensitive domains, and other resource-constrained environments, requiring com-
putational efficiency to coexist with reliability [1, 2]. In this study, SLMs refer
to transformer-based autoregressive language models within the 1B–1.7B pa-
rameter size range [3]. Unlike large cloud-based systems, SLMs often operate
in settings that need uncertainty controlled, hallucination minimized, and gen-
eration stability. However, language models can confidently produce incorrect
outputs, propagate misinformation, and the generation of logically incoherent
output [4–6]. In safety and decision-critical environments, this leads to epistemic
risk. Most evaluation frameworks focus on outcome-related metrics such as ac-
curacy [7], hallucination rate [4], and calibration errors [8]. These are important
metrics. However, they do not explain how the internal model dynamic results
lead to an output. Precisely, analysis that focuses on entropy evolution during
decoding, attention dispersion across layers, and hidden-state representations
remains limited [9]. These comparison at the token and layer levels enables com-
parison between different models for interpretation and systematic regulation
[10].
In transformer models, text generation relies on attention mechanisms and
hidden representations [11]. While prior work has analyzed attention head redun-
dancy and specialization [12], contextual representation geometry [13], structural
hidden state properties [14], uncertainty in language models via entropy and cal-
ibration measures [15], these are done separately. There remains a lack of study
that connects attention and representation dynamics to factual reliability [16].
Entropy is usually treated as a scalar rather than a temporally evolving process
[17]. Consequently, the structure that governs generative truthfulness remains
insufficiently understood, especially for SLMs, where architectural differences
significantly affect model stability.
Therefore, this study introduces a trace-level structural analysis of internal
entropy and attention dynamics in SLMs on TruthfulQA benchmark. By trace-
level structural analysis, we refer to the examination of the decoding steps’ token-
level output entropy, attention entropy, and hidden-state L2 magnitude rather
than the final model correctness. To investigate how internal representation be-
havior maps to uncertainty and truthfulness outcomes, we analyze temporal
entropy transition and cross-metric structural relationships. The study makes 3
major contributions. First, extension of static entropy measurement to a tem-
poral, trace-level framework that captures the evolution of uncertainty during
decoding. Second, probabilistic measures (entropy) are integrated with geomet-
ric measures (hidden-state movement), demonstrating the link between two par-
tially coupled but structurally distinct dimensions. Third, we provide a generic
classification of SLMs behavior (deterministic, exploratory, and balanced) based
on internal uncertainty and stability characteristics. The linkage of truthfulness
evaluation to structural properties advances a uniform framework to analyze
SLMs’ generative reliability, human evaluation, and reliability. This could guide
future design of a more reliable and application-specific edge SLMs.

## Page 3

Contribution Title
3
2
Related works
Truthfulness and hallucination research reveal today’s language model tendency
to generate fluent but factually false responses, especially with misleading or
adversarial prompts [18, 19]. Also, most studies utilize contradiction detection
and accuracy metrics in measuring final output corrections [20, 21] without ex-
plaining the role of decoding dynamics when models fail confidently. This study
performs and analyze token-level entropy and structural behavior to fill this gap.
Numerous existing studies, such as uncertainty estimation, align the model’s
confidence with correctness [22, 23]. In other studies, predictive uncertainty is
measured by Shannon entropy [24, 25], while temperature scaling is explored to
reduce overconfidence [26, 27]. Recent work has applied these techniques [28, 29]
with entropy treated as a single final value. However, none of this study mea-
sures how entropy changes during decoding steps or how these transitions result
in factual reliability.
A wide range of studies have focused on attention mechanisms and inter-
pretability [30, 31], some studies focused on attention heads redundancy and
specializations [32, 33], other studies investigated if attention weights explain
model decisions [34, 35]. Attention layers also encode syntactic and semantic in-
formation [36]. However, there are limited studies that link attention entropy and
output entropy together, which limits understanding of how attention patterns
relate to prediction confidence, which this study fills. Other previous studies on
representation geometry indicate that transformer layers have a structured but
non-uniform transformation [37, 38]. Also, similarity analysis reveals layer spe-
cialization [39], while mechanistic interpretability investigates functional compo-
nents. Nevertheless, Uncertainty or truthfulness is rarely linked with geometric
properties; they are usually separately studied. This work connects representa-
tional drift, hidden-state magnitude, and entropy evolution to better understand
reliability.
Studies have focused on reasoning consistency in language models [40, 41],
but few measure how entropy or attention change over time. Our study models
this across decoding steps to examine structural differences between the mod-
els. Finally, SLMs have been increasingly deployed in resource-constrained or
privacy-sensitive domains [42, 43]. However, reliability research majorly focuses
on bigger models, while smaller models are less examined. Consequently, this
study focuses on smaller models (1B–1.7B parameters). Generally, while previ-
ous work on truthfulness, entropy, attention, and representation geometry has
been conducted independently on larger models, this work integrates them into
a uniform framework linking internal structural dynamics to factual reliability
in SLMs.

## Page 4

4
A. Adeseye et al
3
Experimental Design and Trace-Level Structural
Extraction
3.1
Dataset (TruthfulQA)
This study evaluates model behavior using the TruthfulQA benchmark (con-
tains 790 questions). This is a dataset designed to generate truthful and not just
merely plausible responses. It is appropriate for this study because it spans mul-
tiple areas such as science, health, economics, history, law, and cultural topics,
among other categories as well. It also provides differentiation between correct,
misleading, and partially correct answers, supporting analysis beyond just bi-
nary classifications. All models were prompted with the same set of questions
with greedy decoding to minimize sampling variability. No fine-tuning or cali-
bration was applied, ensuring that results reflect inherent model behavior under
the same default settings. In general, it serves as an examiner of how structural
dynamics differ between truthful and misleading responses.
3.2
Model Selection and Computational Configuration
This study evaluates four transformer-based SLMs within the 1B–1.7B param-
eter range: Llama-3.2-1B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B, Gemma-3-
1B-it, and Qwen3-1.7B. The selection was driven by model family, training, inter-
nal scaling behavior, and architectural differences amongst similar-sized models.
The goal was not to compare model scale but internal structural differences that
influence uncertainty and reliability. Execution was performed on a system with
64GB RAM and 8GB VRAM using FP16. Greedy decoding was intentionally
selected instead of probabilistic sampling to simplify reproducibility. The deter-
ministic rule removes stochastic variability that was introduced via temperature
or nucleus sampling. Consequently, the entropy values reflect intrinsic model
uncertainty rather than just randomness injected during sampling.
3.3
Human Evaluation and Reliability
To determine whether a response is truthful requires human evaluation. All the
generated responses from the model were independently assessed by 2 indepen-
dent researchers who reconciled the output with the TruthfulQA dataset. Both
researchers labeled all responses independently first. Afterwards, disagreements
were reconciled via consensus. The inter-rater reliablity using Cohen’s kappa (κ),
and Krippendorff’s alpha (α). Agreement was strong (Percent Agreement = 88%,
κ = 0.81, α = 0.79), indicating substantial reliability under standard interpreta-
tion guidelines. This confirms the reproducibility and classification consistency of
the process. The results provided the foundation needed to measure the internal
behavior of the models.

## Page 5

Contribution Title
5
Algorithm 1: Trace-Level Structural Extraction for Small Language
Models (SLMs)
Input: Model list M, prompt P , max steps Smax, attention flag A
Output: Per-model trace CSV files and summary CSV
Initialize summary data[”Prompt”] ←P
Set device ←CUDA if available else CPU
Set dtype ←FP16 if CUDA else FP32
foreach m ∈M do
Load tokenizer Tm and model m
Set m to evaluation mode and disable gradients
Build stop token set Stopm using:
tokenizer EOS IDs, config EOS IDs, generation EOS IDs,
common stop tokens, and model-specific identifiers
Build input IDs X using chat template if available;
otherwise tokenize raw prompt
Initialize empty list trace rows
// Prompt Phase (step = 0)
out ←m(X, use cache=True, output hidden states=True, output attentions=A)
Extract layer-wise metrics from out and append to trace rows
Initialize gen ids ←X
Initialize past ←out.past key values
Initialize step ←0
Initialize recent tokens ←[ ]
Initialize collected gen ids ←[ ]
// Generation Phase
while True do
step ←step + 1
out ←m(gen ids[−1], past key values = past, use cache=True,
output hidden states=True, output attentions=A)
xt+1 ←arg max(out.logitst)
Append xt+1 to gen ids and collected gen ids
past ←out.past key values
Extract layer-wise metrics from out and append to trace rows
Update recent tokens (retain last 15 tokens)
if 15 consecutive tokens are identical then
break
if xt+1 ∈Stopm or step ≥Smax then
break
Decode final output from collected gen ids
Store in summary data[m]
Save trace rows to per-model trace CSV
Free model and tokenizer memory; clear cache
Save summary data to summary CSV
3.4
Algorithmic of Trace-Level Extraction
This section contains the algorithm 1 that explains how the model’s internal
behavior was extracted. A more detailed explanation can be seen in the appendix.
4
Entropy-Based Structural Evaluation and Generation
Dynamics
4.1
Cross-Model Structural Entropy and Attention Analysis
Table 1 presents clear structural differences across the four SLMs during the
TruthfulQA generation. The results show distinctively different entropy states,

## Page 6

6
A. Adeseye et al
attention behavior, and decoding confidence. For generated token length (Gen),
Qwen (431 tokens) and LLaMA (360 tokens) produce substantially longer out-
puts than Gemma (133 tokens) on average, while DeepSeek generated the least
(26 tokens). The implication is that longer output increases structural trace
depth. However, it is important to note that KV memory is not purely driven
by generation length but affected by architectural factors such as hidden size
and number of attention heads. This explains why Qwen showed an extremely
large KV footprint despite not having an average lesser token than LLaMA and
slightly more than Gemma and DeepSeek.
Shannon entropy reflects distributional spread (Shannon, 1948). Output en-
tropy is a representation of Shannon entropy. DeepSeek produced highly peaked
probability distributions, while Gemma distributed probability more evenly across
candidate tokens. Gemma had the highest mean output entropy (0.678), while
DeepSeek had the lowest (0.124). A more accurate output is usually associated
with lower output entropy, while an incorrect answer is usually associated with
higher entropy. DeepSeek was the most highly confident model (strong inter-
nal certainty) because of its low entropy, which aligns with a very high Top1
probability (0.970) and a large Top1–Top2 gap (0.957). Gemma and Qwen were
moderately confident, while LLaMA sits in between. Also, output entropy dis-
tribution analysis indicates that for LLaMA and Gemma, the mean exceeds
the median, which means most tokens had low entropy but occasionally exhibit
very high uncertainty. Deepseek had the tightest distribution, where most of
its tokens are almost deterministic. Qwen falls in the middle of both of these
categories. Output entropy standard deviation further reflects model stability.
Gemma (0.920) had the highest volatility, which suggests an exploratory decod-
ing behavior, then LLaMA (0.722) and Qwen (0.527). DeepSeek (0.483) was the
most stable
For attention entropy, Gemma had the highest mean attention entropy (2.402),
LLaMA was moderate, while DeepSeek was the lowest. Usually, high attention
points to diffused contextual focus, while lower attention denotes a more con-
centrated focus. Also, attention weights define how contextual information is
integrated. Consequently, diffuse attention means support for broader evidence
aggregation, whereas focused attention may reflect stronger token-level selec-
tivity. This means Gemma generally had a broader scanning strategy while
Deepseek generally had a narrower mean attention focus. Standard deviation
measures stability during decoding steps. DeepSeek had the lowest SD (0.139),
indicating highly stable attention behavior. Gemma (0.255) is moderately sta-
ble, followed by LLaMA (0.292), which exhibited slightly more fluctuation, while
Qwen had the highest SD (0.325). This means DeepSeek’s attention structure
did not change very much across steps, while Qwen’s had the most dynamic fluc-
tuations. Exhibiting stable attention suggests consistent structural allocation of
contextual importance, whereas higher variability may indicate adaptive context
shifting. The relationship between mean and median further clarifies distribu-
tional shape. For all models, mean ≈median, which indicates that attention
entropy distributions are roughly symmetric and stable, unlike output entropy,

## Page 7

Contribution Title
7
which exhibited strong right skew. Therefore, attention behavior did not exhibit
extreme spikes in the same way that token-level output entropy uncertainty did.
According to multi-head attention theory, higher dispersion may indicate
stronger head specialization and representational diversity [44]. Head Dispersion
Index (HDI) reveals significant architectural variation [45]. Qwen (0.951) had
the highest dispersion, pointing to a more uniform attention heads behavior.
LLaMA was also considerably high. Contrariwise, DeepSeek and Gemma had
lower HDI values, which reflect a more uniform head behavior.
Table 1. Overall Generation and Entropy Summary (GEN Phase). Values are reported
as Mean (SD). Entropy measures represent token-level Shannon entropy and are non-
negative by definition.
Metric
LLaMA-3.2-1B Gemma-3-1B DeepSeek-1.5B
Qwen-1.7B
Prompt Tokens
46
19
16
23
Gen Tokens
360
133
26
431
Output Entropy
0.570 ± 0.722
0.678 ± 0.920
0.124 ± 0.483
0.374 ± 0.527
Top1 Prob
0.817 ± 0.225
0.806 ± 0.220
0.970 ± 0.111
0.875 ± 0.182
Top1–Top2 Gap
0.713
0.697
0.957
0.795
Attn Entropy Mean
1.809 ± 0.292
2.402 ± 0.255
1.896 ± 0.139
1.966 ± 0.325
Head Dispersion (HDI)
0.871 ± 0.128
0.517 ± 0.085
0.495 ± 0.078
0.951 ± 0.131
KV Total MB (Max)
25.375
7.719
2.297
99.312
NOTE: Values are reported as Mean (SD), where the number in parentheses denotes the standard de-
viation across generated tokens. Output Entropy and Attention Entropy represent token-level Shannon
entropy (H = −P
p(x) log p(x)) and are non-negative by definition. Top1 Prob refers to the mean
probability assigned to the selected next token. Top1–Top2 Gap denotes the average difference between
the highest and second-highest token probabilities. HDI represents the Head Dispersion Index. KV Total
MB indicates the maximum key-value cache memory usage during generation.
Table 2 presents the other output entropy distribution profile not present in
Table 1.
All models had P10 = 0 and Min = 0, indicating that at least 10% of to-
kens were fully deterministic. All models exhibited these deterministic predic-
tions during token generation. However, P90 values showed clear divergence.
LLaMA (1.571) and Gemma (1.472) frequently entered high-uncertainty states,
with Qwen (1.079) being moderate and DeepSeek (0.018) being extremely low.
This means 90% of DeepSeek tokens are almost deterministic, whereas LLaMA
and Gemma occasionally had strong internal competition between candidate to-
kens. Usually, high P90 values indicate unstable or exploratory moments in the
decoding trajectory.
Maximum entropy helps capture rare extreme events. Gemma had the highest
maximum (5.662), followed by LLaMA (3.706) and Qwen (2.947). DeepSeek
had the lowest (2.376). This means Gemma has the highest uncertainty tail; it
occasionally produces extremely uncertain tokens. In factual benchmarks like
TruthfulQA, such spikes may correspond to epistemic confusion or knowledge
boundary cases.

## Page 8

8
A. Adeseye et al
Generally, three behavioural insights emerged from the distributional struc-
ture. DeepSeek operates in a highly deterministic manner, characterized by very
low median entropy, very low P90, and a tight distribution. Gemma operates in
a volatile or exploratory manner, with the highest mean entropy, highest SD,
and strongest extreme spikes. LLaMA and Qwen operate in a balanced manner,
with moderate entropy levels and moderate volatility.
From a theoretical point of view, entropy measures uncertainty in proba-
bilistic prediction [46]. Consequently, very low entropy is a reflection of rigid
confidence, which may increase the risk of confident errors. Also, very high en-
tropy reflects unstable reasoning and token competition. Balanced entropy with
controlled variance may support better calibration and structured reasoning dur-
ing uncertainty. These distributional patterns provide structural insight into how
SLMs manage epistemic uncertainty during TruthfulQA generation.
Table 2. Other Output Entropy Distribution Profile
Statistic LLaMA-1B Gemma-1B DeepSeek-1.5B Qwen-1.7B
P10
0.000
0.000
0.000
0.000
P90
1.571
1.472
0.018
1.079
Min
0.000
0.000
0.000
0.000
Max
3.706
5.662
2.376
2.947
Table 3 presents the other parameters from the distribution of step-wise
attention entropy across models not found in Table 1.
Percentile ranges confirm these structural patterns. Qwen shows a relatively
wide P10–P90 range (1.475 →2.347), indicating noticeable variability in atten-
tion dispersion across steps. DeepSeek shows a very tight range (1.759 →1.998).
This confirms that DeepSeek maintains a highly consistent attention structure
throughout generation. Gemma maintains high entropy even at P10 (2.028),
meaning even its lowest attention states remain relatively diffuse.
Minimum and maximum values reinforce these trends. Gemma reaches the
highest maximum (2.995), suggesting occasional very broad context integration.
DeepSeek operates within the narrowest band. From a theoretical standpoint,
attention entropy captures distributional smoothness of attention weights. Con-
trolled and stable attention may reduce structural noise, while excessive disper-
sion may dilute representational strength (Michel et al., 2019).
Overall, three attention regimes emerge. Gemma operates in a high-dispersion
regime with consistently diffuse attention. DeepSeek operates in a concentrated
and highly stable regime. LLaMA and Qwen sit in a moderate regime, with
Qwen showing more fluctuation across steps. These findings indicate that at-
tention dynamics are architecturally structured and differ independently from
output entropy behavior. Such structural attention differences may influence how
models integrate contextual evidence when responding to TruthfulQA prompts.

## Page 9

Contribution Title
9
Table 3. Step-Wise Attention Entropy Distribution
Statistic LLaMA-1B Gemma-1B DeepSeek-1.5B Qwen-1.7B
P10
1.387
2.028
1.759
1.475
P90
2.173
2.708
1.998
2.347
Min
0.994
1.712
1.647
1.094
Max
2.475
2.995
2.335
2.719
Layer-Level Attention Structure Table 4 presents the layer-wise entropy
profile across models. This analysis reveals how attention dispersion behaves
vertically across transformer depth.
The number of layers differs across architectures. LLaMA has 16 layers,
while Gemma has 26, and DeepSeek and Qwen each have 28. This matters
because deeper models can distribute representational roles differently across
layers. Greater depth allows hierarchical abstraction and progressive refinement
of information. In transformer theory, early layers often capture local relations,
while deeper layers encode higher-level semantics [46].
Mean layer entropy follows the global pattern observed earlier. Gemma main-
tains the highest average layer entropy (2.402), indicating consistently diffuse
attention across depth. LLaMA has the lowest mean (1.809), suggesting more
concentrated average attention. DeepSeek and Qwen sit between these extremes.
Structural imbalance across layers is captured by the standard deviation
across layers. Qwen shows the highest SD (0.666), followed by LLaMA (0.600),
Gemma (0.537), and DeepSeek (0.351). This shows that Qwen and LLaMA have
stronger layer-wise variation, while DeepSeek is structurally more uniform across
depth. Higher cross-layer variance suggests that different layers perform sharply
distinct functions. Lower variance suggests smoother vertical transitions.
The gap between lowest and highest entropy layers highlights specialization.
Qwen’s lowest entropy layer is 0.836, while its highest is 3.153. This is a very
large internal range. It indicates strong specialization — some layers are highly
focused, while others are highly diffuse. In contrast, DeepSeek’s range is much
tighter. This confirms that Qwen exhibits strong vertical heterogeneity, while
DeepSeek maintains consistent structural behavior across depth.
Mean HDI further reinforces this pattern. Qwen shows the highest HDI
(0.951), followed by LLaMA (0.871). Gemma (0.517) and DeepSeek (0.495) are
substantially lower. This means Qwen has the most uneven head behavior within
layers, suggesting stronger head specialization. DeepSeek and Gemma show more
uniform head behavior. According to multi-head attention theory, specialization
across heads supports representational diversity [44]. However, excessive hetero-
geneity may also increase structural instability.
When combining step-level and layer-level findings, two structural dimen-
sions become visible. At the step level, Gemma is the most diffuse overall, while
DeepSeek is the most stable across decoding steps. At the layer level, Qwen is
the most structurally heterogeneous, while DeepSeek is the most structurally

## Page 10

10
A. Adeseye et al
uniform. This indicates that architectural depth and head dispersion introduce
a second axis of structural differentiation beyond token-level entropy.
From a theoretical perspective, transformers construct hierarchical internal
representations across depth [19]. Layer-wise entropy variation reflects how at-
tention shifts from broad contextual encoding to refined information compres-
sion. Uniform vertical structure, as seen in DeepSeek, suggests controlled ab-
straction. Strong vertical heterogeneity, as seen in Qwen, suggests dynamic spe-
cialization across layers. These structural properties provide insight into how
architectural depth shapes internal reasoning organization during TruthfulQA
generation.
Table 4. Layer-Wise Entropy Profile
Metric
LLaMA-1B Gemma-1B DeepSeek-1.5B Qwen-1.7B
#Layers
16
26
28
28
Mean Layer Entropy
1.809
2.402
1.896
1.966
SD Across Layers
0.600
0.537
0.351
0.666
Lowest Entropy Layer
L3 (1.015)
L18 (1.313)
L1 (1.196)
L25 (0.836)
Highest Entropy Layer
L1 (2.787)
L9 (3.048)
L8 (2.447)
L2 (3.153)
Mean HDI
0.871
0.517
0.495
0.951
4.2
Temporal Entropy Dynamics and Representational Evolution
Early vs Late Generation Entropy Drift Table 5 examines how entropy
changes from the first 20% to the last 20% of generation. The results show clear
temporal transitions in both attention and output uncertainty.
Attention entropy increases for all models. LLaMA (+0.666), Gemma (+0.568),
DeepSeek (+0.284), and Qwen (+0.715) all show positive shifts. This means at-
tention becomes more diffuse toward the end of generation for every model.
As generation progresses, attention spreads out more. This is a universal trend
across architectures. In autoregressive transformers, later tokens must integrate
longer context windows (Vaswani et al., 2017). Broader attention may reflect
increased contextual aggregation.
Output entropy, however, splits into two distinct behaviours. LLaMA (-0.388)
and DeepSeek (-0.526) show decreasing output entropy. This means these models
become more confident over time. Their probability distributions sharpen, and
they “lock in” toward the end of generation. DeepSeek shows the most extreme
case. Early output entropy is 0.528, while late entropy drops to 0.002. This is
almost deterministic. It suggests DeepSeek strongly commits to final tokens.
In contrast, Gemma (+0.489) and Qwen (+0.255) show increasing output
entropy. This means these models become more uncertain later in generation.
They shift into a more exploratory regime rather than consolidating confidence.
This pattern may relate to factual instability in longer responses. As sequence

## Page 11

Contribution Title
11
length increases, uncertainty may accumulate if internal representations are not
tightly stabilized.
Importantly, attention entropy increases for all models, but output entropy
does not follow the same direction. This means internal attention diffusion does
not directly determine prediction confidence. Attention dispersion and token-
level certainty are related but not identical processes. Attention governs infor-
mation integration, while output entropy reflects final probability allocation over
vocabulary space.
From an information-theoretic perspective, entropy drift reflects dynamic
changes in epistemic uncertainty (Shannon, 1948). Decreasing entropy suggests
convergence toward a stable belief state. Increasing entropy suggests growing
ambiguity or representational diffusion.
Overall, entropy is not static during generation. Models transition into differ-
ent uncertainty regimes. Some models consolidate confidence over time (DeepSeek,
LLaMA). Others amplify uncertainty (Gemma, Qwen). This provides strong ev-
idence that generative behaviour differs dynamically across architectures. These
temporal differences highlight that architectural design influences not only static
entropy levels but also how uncertainty evolves during TruthfulQA generation.
Table 5. Early vs Late Entropy Shift (First 20% vs Last 20%)
Metric
LLaMA-1B Gemma-1B DeepSeek-1.5B Qwen-1.7B
Output Entropy (Early)
0.865
0.508
0.528
0.245
Output Entropy (Late)
0.477
0.996
0.002
0.500
∆Output
−0.388
0.489
−0.526
0.255
Attn Entropy (Early)
1.453
2.053
1.766
1.533
Attn Entropy (Late)
2.119
2.621
2.050
2.248
∆Attn
0.666
0.568
0.284
0.715
Extremal Entropy Layers Table 6 compares the lowest and highest entropy
layers in LLaMA-1B together with their corresponding HDI values. The gap
between the lowest entropy layers (1.015–1.378) and the highest entropy lay-
ers (2.321–2.787) is large, approximately 1.7. This is a substantial difference.
It shows that some layers are strongly focused while others are highly diffuse.
LLaMA does not treat all layers equally.
Early layers tend to be more diffuse. Layer 1 has the highest entropy (2.787),
and several early–mid layers (7–10) also appear among the highest entropy group.
This suggests that early and middle layers distribute attention broadly. They
likely function as contextual information gatherers. In transformer theory, lower
layers often encode broad lexical and positional relationships before deeper ab-
straction occurs [47].
Focused layers appear in mid and deeper positions. Layers 3 and 2, as well as
deeper layers such as 14, 13, and 16, show low entropy values. This suggests that

## Page 12

12
A. Adeseye et al
some deeper layers become more concentrated. These layers may refine repre-
sentations and consolidate information before final prediction. Such refinement
behavior aligns with hierarchical representation theory in deep transformers.
HDI patterns further reveal head-level structure. The highest entropy layer
(Layer 1) also has a very high HDI (1.099). This means that in diffuse lay-
ers, attention heads behave unevenly. Some heads dominate or specialize dif-
ferently from others. In contrast, lower entropy layers show lower HDI values
(0.708–0.922). This suggests that focused layers tend to have more uniform head
behavior.
Together, these results demonstrate structural specialization. Entropy is not
evenly distributed across layers. Some layers act as “broad context collectors,”
while others function as “refinement or decision layers.” This reflects architec-
tural specialization within LLaMA. According to multi-head attention theory,
such differentiation supports representational diversity and hierarchical abstrac-
tion (Michel et al., 2019). The layer-wise entropy structure therefore provides
clear evidence of vertical functional differentiation during generation.
Table 6. Comparison of Lowest and Highest Entropy Layers with Corresponding HDI
for LLaMA-1B
Lowest Entropy Layers Highest Entropy Layers
Layer Entropy
HDI
Layer Entropy
HDI
3
1.015
0.821
1
2.787
1.099
2
1.162
0.922
8
2.741
0.932
14
1.171
0.766
9
2.638
0.882
13
1.352
0.708
7
2.380
0.881
16
1.378
0.728
10
2.321
0.989
Representation Magnitude and Transformation Table 7 reports the mean
L2 norm of hidden representations and the average step-to-step change (Delta
L2). These metrics describe the geometric magnitude of internal states and how
strongly they transform during autoregressive decoding.
The scale differences are extremely large. Gemma-1B shows a mean hid-
den L2 of 6419.709, Qwen-1.7B shows 567.026, DeepSeek-1.5B shows 137.372,
and LLaMA-1B shows 13.132. These values are not just different — they are
on completely different scales. This indicates that hidden state magnitude is
architecture-dependent and cannot be directly compared in absolute terms across
models. Such differences reflect hidden dimension size, normalization strategy
(e.g., RMSNorm vs LayerNorm), and internal scaling choices in model design.
The L2 norm therefore captures representational energy rather than confidence.
Representation drift (Delta L2) shows how much hidden states move be-
tween consecutive tokens. Gemma again shows the largest drift (1600.649), fol-
lowed by Qwen (303.196), DeepSeek (67.798), and LLaMA (9.735). This means

## Page 13

Contribution Title
13
Gemma transforms its hidden state much more aggressively between tokens,
while LLaMA changes its representation more gently and smoothly. Larger drift
suggests stronger internal reconfiguration per decoding step.
When connected to earlier entropy results, an important pattern emerges.
Gemma, which showed higher output entropy and higher attention entropy, also
exhibits the largest representational drift. This suggests that higher uncertainty
is associated with larger internal state transformations. In contrast, DeepSeek,
which showed very low output entropy and strong late-stage confidence con-
solidation, has much smaller drift compared to Gemma. This indicates tighter
representational control.
However, representation magnitude and entropy are not identical constructs.
Large hidden L2 does not automatically imply high entropy. Drift magnitude
reflects transformation strength, not uncertainty directly. Internal geometric en-
ergy and probabilistic uncertainty operate as partially independent dimensions.
Entropy measures distributional uncertainty over tokens, while L2 norms reflect
vector magnitude in hidden space. One captures probabilistic dispersion; the
other captures geometric dynamics.
Overall, the models operate in distinct representational regimes. LLaMA
shows low-magnitude, low-drift dynamics with stable evolution. DeepSeek shows
moderate magnitude and controlled drift. Qwen shows stronger movement. Gemma
operates in a high-magnitude, high-drift regime. These geometric differences re-
inforce that generative behavior is shaped not only by entropy and attention
structure but also by the internal geometry of hidden state evolution.
Table 7. Hidden Representation Magnitude and Drift
Model
Hidden L2 (Mean ± SD) Delta L2 from Prev (Mean ± SD)
LLaMA-1B
13.132 ± 0.362
9.735 ± 0.183
Gemma-1B
6419.709±738.524
1600.649±148.048
DeepSeek-1.5B
137.372 ± 6.277
67.798 ± 2.762
Qwen-1.7B
567.026±41.272
303.196±31.500
5
Discussion
The figure 1 presents component trends across four small-scale language mod-
els (LLaMA-1B, Gemma-1B, DeepSeek-1.5B, and Qwen-1.7B) under four out-
come types: Best Answer, Correct Answer, Best Incorrect, and Incorrect. Per-
formance is evaluated using four metrics: Identified, Wrongly Classified, Hallu-
cination, and Accurately Classified. Across all models, the number of accurately
classified and identified instances increases from correct to incorrect categories,
while hallucination rises sharply in the Best Incorrect and Incorrect outcomes.
Wrong classifications remain relatively low and stable compared to hallucina-
tion. DeepSeek-1.5B and Gemma-1B show stronger gains in identification and

## Page 14

14
A. Adeseye et al
accurate classification in the incorrect categories, whereas LLaMA-1B and Qwen-
1.7B demonstrate more gradual trends. The pattern suggests that hallucination
is more strongly triggered in error-prone responses, while accurate classification
correlates with stronger answer quality. Importantly, all classifications (identi-
fied, accurately classified, wrongly classified, and hallucinated instances) were
manually verified and cross-checked with the researchers to ensure consistency,
validity, and reliability of the evaluation.
Fig. 1. Component Trends Across Outcome Types (Per Model)
Figure 2 shows that entropy evolves during generation. Attention entropy in-
creases for all models, meaning attention becomes more diffuse toward the end of
responses. However, output entropy does not follow the same pattern. DeepSeek-
1.5B and LLaMA-1B reduce output entropy over time, indicating stronger con-
fidence consolidation. In contrast, Gemma-1B and Qwen-1.7B increase output
entropy in later stages, indicating growing uncertainty. Attention diffusion and
prediction confidence are related but not identical processes. Models can broaden
attention while simultaneously sharpening or weakening token-level certainty.

## Page 15

Contribution Title
15
Fig. 2. Entrophy Drift
Figure 3 (Left top image) highlights a structural trade-off between entropy
and Top1 probability. DeepSeek operates in a low-entropy, high-confidence regime.
Gemma operates in a high-entropy, lower-confidence regime. LLaMA and Qwen
fall between these extremes. When linked to Table 1, these regimes align with
behavioural outcomes. DeepSeek achieves the highest proportion of accurately
classified Best Answers (21%), while Gemma shows the highest hallucination
proportions in incorrect categories. LLaMA and Qwen maintain more balanced
profiles. Extreme determinism and extreme exploration both carry risks; cali-
brated confidence appears more beneficial than either extreme.
Figure 3 (right top image) shows that KV memory footprint does not di-
rectly determine entropy behavior. Qwen uses the highest memory but does not
achieve the lowest uncertainty. DeepSeek uses minimal memory yet maintains
strong decisiveness. Architectural design influences uncertainty more than raw
memory usage. Structural efficiency and probabilistic control operate in partially
independent dimensions.

## Page 16

16
A. Adeseye et al
Fig. 3. Relationships Between Output Entropy, Top-1 Confidence, KV Memory, Rep-
resentational Drift, and Attention Dispersion Across Small Language Models
Figure 3 (left bottom image) reveals a clear relationship between hidden-
state drift (Delta L2) and output entropy. Gemma exhibits the largest repre-
sentational drift and the highest entropy. DeepSeek shows the smallest drift
and the lowest entropy. LLaMA and Qwen lie between these extremes. Stronger
geometric transformations across decoding steps are associated with higher un-
certainty. However, drift magnitude is not the same as entropy. Drift reflects
geometric change in hidden space, while entropy reflects probabilistic dispersion
over tokens. Geometric energy and probabilistic uncertainty are coupled but
conceptually distinct.
Figure 3 (right bottom image) shows that attention head specialization varies
across architectures. Qwen has the highest Head Dispersion Index (HDI), indi-
cating strong head heterogeneity. DeepSeek and Gemma show more uniform
head behavior. Combined with earlier layer-level analysis, this confirms struc-
tural specialization across depth. Some layers act as broad context integrators,
while others perform refinement and consolidation. Head diversity and entropy
are related but not perfectly aligned, indicating that architectural specialization
shapes reasoning pathways.
Figure 4 integrates all structural measures. Output entropy shows a strong
negative correlation with Top1 probability, confirming the confidence–uncertainty
relationship. Entropy positively correlates with representation drift, supporting
the link between geometric instability and probabilistic uncertainty. Attention

## Page 17

Contribution Title
17
entropy shows moderate association with output entropy, while KV memory
shows weak direct correlation. No single structural metric explains truthfulness
on its own; truthfulness emerges from interacting structural components.
Fig. 4. Correlation heatmap of entropy, confidence, attention dispersion, representation
drift, and memory metrics.
Table 1 connects these structural findings to behavioral outcomes. DeepSeek
represents a deterministic regime with low entropy, stable geometry, and strong
confidence consolidation. Gemma represents an exploratory regime with high
entropy, large drift, and higher hallucination rates. LLaMA and Qwen represent
balanced regimes with moderate entropy and moderate stability. Truthfulness is
therefore not driven by model size alone but by how uncertainty evolves, how
representations shift, and how attention is distributed during decoding.
The high inter-rater agreement strengthens these conclusions. With κ = 0.81
and α = 0.79, the outcome classifications are reliable and not driven by subjec-
tive bias. Disagreements mainly occurred in borderline distinctions between Best
and Correct answers, while severe hallucinations showed near-perfect agreement.
This confirms that the structural–behavioral relationships observed are robust
and not annotation artifacts.
Overall, the results demonstrate that truthfulness is a structural phenomenon
emerging from the interaction between entropy evolution, attention diffusion,
head specialization, and representational stability. Accuracy alone cannot cap-
ture this complexity. Generative reliability depends on how probabilistic uncer-
tainty and geometric dynamics are internally regulated over time.
6
Conclusion
This study introduced a trace-level structural framework to analyze generative
reliability in SLMs. Instead of evaluating only final output accuracy, internal
decoding dynamics across token-level entropy, attention diffusion, head disper-
sion, hidden-state magnitude, and representational drift were analyzed. By ex-

## Page 18

18
A. Adeseye et al
tracting structural signals at each decoding step and layer, we moved beyond
outcome-level evaluation to provide insight into how uncertainty evolves during
generation.
Our analysis reveals that entropy regulation is not static. Attention diffusion
increased across all models during decoding, yet prediction confidence evolved
differently across architectures. Some models showed decreased output entropy
over time, indicating consolidation toward deterministic prediction, while oth-
ers exhibited increasing entropy, reflecting exploratory behavior. Hidden-state
magnitude and layer-to-layer representational drift further demonstrated that
geometric transformation and probabilistic uncertainty operate as partially inde-
pendent dimensions. These findings suggest that truthfulness is not determined
primarily by confidence levels but emerges from dynamic interactions between
entropy regulation, attention allocation, and representational geometry.
The integration of probabilistic and geometric analysis contributes a struc-
tural methodology for evaluating generative stability in SLMs. By distinguishing
deterministic, exploratory, and balanced structural patterns, this study provides
a principled framework for understanding architectural differences in SLMs. For
future work. First, the study uses greedy decoding; extending to temperature
or nucleus sampling would test how structural entropy interacts with sampling
variability. Second, moving beyond TruthfulQA to multi-hop and long-form rea-
soning would examine behavior under deeper cognitive load. Third, analyzing
step-to-step representational drift across tokens not only layer-wise drift could
expose temporal instability in reasoning chains. Fourth, scaling to larger mod-
els would clarify how size affects entropy regulation and geometric stability. Fi-
nally, structural metrics can be embedded into adaptive, entropy-aware decoding
and attention-regularized training, enabling uncertainty-guided optimization for
building more reliable SLMs by directly controlling internal dynamics rather
than only improving final accuracy.
By shifting the focus from output-only evaluation to trace-level structural
dynamics, this work advances a more interpretable and principled understanding
of generative reliability in SLMs.
7
Declaration on the Use of Generative AI
Language editing and grammar-checking tools were used to improve clarity and
readability of the manuscript.
Appendix
8
Algorithm Description of Trace-Level Extraction
8.1
Prompt Standardization and Input Construction
Instruction-tuned models require consistent formatting for predictable behavior.
Some models use chat-style templates, while others accept raw text input. To en-
sure consistency, the tokenizer was inspected for the presence of a chat template.

## Page 19

Contribution Title
19
If available, the prompt was wrapped using the tokenizer’s apply chat template
function. Otherwise, the prompt was tokenized directly. This normalization en-
sures models receive input in the format used during training. Without such
standardization, formatting differences could artificially influence entropy and
attention behavior. The procedure therefore isolates architectural differences
rather than prompt-formatting effects.
8.2
Decoding Protocol and Termination Strategy
Generation was performed step-by-step using autoregressive decoding with key–
value caching. After processing the full prompt, cached key and value tensors
were stored for each transformer layer. During generation, only the most recently
generated token was passed back into the model along with cached states, pre-
serving full historical context while reducing computational overhead. Decoding
terminated under three conditions:
1. If a model-specific stop token was generated. Stop tokens were constructed
by combining tokenizer-defined EOS tokens, configuration identifiers, genera-
tion configuration tokens, and common special tokens (e.g., <eos>, <endoftext>).
2. If a maximum of 1000 tokens was reached.
3. If the same token was generated for 15 consecutive steps, preventing degen-
erate repetition.
These termination rules ensure fairness across architectures and prevent
pathological generation behavior.
8.3
Trace-Level Structural Signal Extraction
Instead of evaluating only the final generated output, internal signals were
recorded at every decoding step and for every transformer layer. Extraction
occurred in two phases:
Prompt Phase (Step 0): A full forward pass over the input prompt was
executed. Hidden states, attention tensors, logits, and key–value cache contents
were extracted.
Generation Phase: The same signals were extracted at each decoding step.
Each step produced hidden states corresponding to the embedding output and
each transformer layer, along with attention tensors when enabled. Each de-
coding step generated multiple rows in the trace file—one per layer plus the
embedding output—forming a long-format dataset indexed by phase, step, and
layer.
8.4
Probabilistic Uncertainty Measurement
At each decoding step, logits were converted into probabilities using the softmax
function. Token-level Shannon entropy was computed as:

## Page 20

20
A. Adeseye et al
H = −
X
i
pi log pi
(1)
Low entropy indicates high confidence, while high entropy reflects uncer-
tainty. Two additional confidence measures were recorded:
– Top-1 probability
– Top-1–Top-2 probability gap
The Top-1–Top-2 gap measures decisiveness. A larger gap indicates stronger
preference for a single token.
8.5
Attention Entropy and Diffusion
For each transformer layer, attention weights corresponding to the last query po-
sition were extracted and normalized. Entropy was computed across attended key
positions for each attention head. The mean entropy across heads was recorded
as layer-level attention entropy. Lower attention entropy indicates concentrated
attention, while higher entropy indicates diffusion.
8.6
Hidden-State Magnitude and Representation Drift
For each layer, the L2 norm of the last-token hidden state was computed. Rep-
resentation drift between consecutive layers was measured as:
∆l = ∥hl −hl−1∥2
(2)
Larger values indicate stronger representational transformation. Entropy cap-
tures probabilistic uncertainty, whereas representation drift captures geometric
movement in vector space. These dimensions are related but distinct.
8.7
Key-Value Cache Memory Measurement
Memory usage was computed by measuring the byte size of key and value tensors
stored in the cache for each layer. Both per-layer and total memory usage were
recorded. This enables structural scaling differences to be observed as sequence
length increases and provides insight into architectural efficiency beyond entropy
analysis.
References
1. Sharshar, A., Khan, L.U., Ullah, W., Guizani, M.: Vision-language models for edge
networks: a comprehensive survey. IEEE Internet Things J. 12(16), 32701–32724
(2025). doi:10.1109/JIOT.2025.3579032
2. Qu, G., Chen, Q., Wei, W., Lin, Z., Chen, X., Huang, K.: Mobile edge intelligence
for large language models: a contemporary survey. IEEE Commun. Surv. Tutor.
27(6), 3820–3860 (2025). doi:10.1109/COMST.2025.3527641

## Page 21

Contribution Title
21
3. Wang, F., Lin, M., Ma, Y., Liu, H., He, Q., Tang, X., Tang, J., Pei, J., Wang, S.:
A survey on small language models in the era of large language models: architec-
ture, capabilities, and trustworthiness. In: Proceedings of the 31st ACM SIGKDD
Conference on Knowledge Discovery and Data Mining (KDD ’25), pp. 6173–6183.
ACM, New York (2025). doi:10.1145/3711896.3736563
4. Adeseye, A., Isoaho, J., Tahir, M.: Performance evaluation of LLM hallucination
reduction strategies for reliable qualitative analysis. In: Arabnia, H.R., Deligian-
nidis, L., Amirian, S., Ghareh Mohammadi, F., Shenavarmasouleh, F. (eds.) AI
Revolution: Research, Ethics and Society, pp. 142–156. Springer Nature Switzer-
land, Cham (2026). doi:10.1007/978-3-032-12313-8 11
5. Augenstein, I., Baldwin, T., Cha, M., Chakraborty, T., Ciampaglia, G.L., Corney,
D., DiResta, R., Ferrara, E., Hale, S., Halevy, A., Hovy, E., Ji, H., Menczer, F.,
Miguez, R., Nakov, P., Scheufele, D., Sharma, S., Zagni, G.: Factuality challenges
in the era of large language models and opportunities for fact-checking. Nat. Mach.
Intell. 6(8), 852–863 (2024). doi:10.1038/s42256-024-00881-z
6. Shah, S.B., Thapa, S., Acharya, A., Rauniyar, K., Poudel, S., Jain, S., Masood,
A., Naseem, U.: Navigating the web of disinformation and misinformation: large
language models as double-edged swords. IEEE Access 13, 169262–169282 (2025).
doi:10.1109/ACCESS.2024.3406644
7. Adeseye, A., Isoaho, J., Tahir, M.: Systematic prompt framework for qualitative
data analysis: designing system and user prompts. In: 2025 IEEE 5th International
Conference on Human-Machine Systems (ICHMS), pp. 229–234. IEEE (2025). doi:
10.1109/ICHMS65439.2025.11154183
8. Xia, Y., Luz De Araujo, P.H., Zaporojets, K., Roth, B.: Influences on LLM cal-
ibration: a study of response agreement, loss functions, and prompt styles. In:
Che, W., Nabende, J., Shutova, E., Pilehvar, M.T. (eds.) Proceedings of the
63rd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), pp. 3740–3761. Association for Computational Linguistics (2025).
doi:10.18653/v1/2025.acl-long.188
9. Skean, O., Arefin, M.R., Zhao, D., Patel, N., Naghiyev, J., LeCun, Y., Shwartz-Ziv,
R.: Layer by layer: uncovering hidden representations in language models. arXiv
preprint arXiv:2502.02013 (2025). https://arxiv.org/abs/2502.02013
10. Apidianaki, M.: From word types to tokens and back: a survey of approaches to
word meaning representation and interpretation. Comput. Linguist. 49(2), 465–523
(2023). doi:10.1162/coli a 00474
11. Vig, J., Belinkov, Y.: Analyzing the structure of attention in a transformer lan-
guage model. In: Linzen, T., Chrupala, G., Belinkov, Y., Hupkes, D. (eds.) Pro-
ceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting
Neural Networks for NLP, pp. 63–76. Association for Computational Linguistics
(2019). doi:10.18653/v1/W19-4808
12. Voita, E., Talbot, D., Moiseev, F., Sennrich, R., Titov, I.: Analyzing multi-head
self-attention: specialized heads do the heavy lifting, the rest can be pruned. In:
Korhonen, A., Traum, D., Marquez, L. (eds.) Proceedings of the 57th Annual Meet-
ing of the Association for Computational Linguistics, pp. 5797–5808. Association
for Computational Linguistics (2019). doi:10.18653/v1/P19-1580
13. Li, Z., Cen, J., Su, B., Huang, W., Xu, T., Rong, Y., Zhao, D.: Large language-
geometry model: when LLM meets equivariance. arXiv preprint arXiv:2502.11149
(2025). https://arxiv.org/abs/2502.11149
14. Servedio, G., De Bellis, A., Di Palma, D., Anelli, V.W., Di Noia, T.: Are the hidden
states hiding something? Testing the limits of factuality-encoding capabilities in

## Page 22

22
A. Adeseye et al
LLMs. In: Che, W., Nabende, J., Shutova, E., Pilehvar, M.T. (eds.) Proceedings of
the 63rd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), pp. 6089–6104. Association for Computational Linguistics (2025).
doi:10.18653/v1/2025.acl-long.304
15. Liu, X., Chen, T., Da, L., Chen, C., Lin, Z., Wei, H.: Uncertainty quantification
and confidence calibration in large language models: a survey. In: Proceedings of
the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining
(KDD ’25), pp. 6107–6117. ACM, New York (2025). doi:10.1145/3711896.3736569
16. Wang, W., Haddow, B., Birch, A., Peng, W.: Assessing factual reliability of large
language model knowledge. In: Duh, K., Gomez, H., Bethard, S. (eds.) Pro-
ceedings of the 2024 Conference of the North American Chapter of the Asso-
ciation for Computational Linguistics: Human Language Technologies (Volume
1: Long Papers), pp. 805–819. Association for Computational Linguistics (2024).
doi:10.18653/v1/2024.naacl-long.46
17. Zhu, C., Wu, S., Zeng, X., Xu, Z., Kang, Z., Guo, Y., Lu, Y., Huang, J.,
Zhou, G.: EDIS: diagnosing LLM reasoning via entropy dynamics. arXiv preprint
arXiv:2602.01288 (2026). https://arxiv.org/abs/2602.01288
18. Brunello, N.: Trustworthiness of large language models: hallucinations. In: Pil-
lai, A.S., Tedesco, R., Scotti, V. (eds.) Challenges and Applications of Gen-
erative Large Language Models, pp. 107–126. Morgan Kaufmann (2026). doi:
10.1016/B978-0-443-33592-1.00007-3
19. Ahmadi, A.: Unravelling the mysteries of hallucination in large language mod-
els: strategies for precision in artificial intelligence language generation. Asian J.
Comput. Sci. Technol. 13(1), 1–10 (2024). doi:10.70112/ajcst-2024.13.1.4144
20. Galitsky, B., Chernyavskiy, A., Ilvovsky, D.: Truth-O-Meter: handling multiple
inconsistent sources repairing LLM hallucinations. In: Proceedings of the 47th
International ACM SIGIR Conference on Research and Development in Informa-
tion Retrieval (SIGIR ’24), pp. 2817–2821. ACM, New York (2024). doi:10.1145/
3626772.3657679
21. Chandler, A., Surve, D., Su, H.: Detecting errors through ensembling prompts
(DEEP): an end-to-end LLM framework for detecting factual errors. In: Al-
Onaizan, Y., Bansal, M., Chen, Y.-N. (eds.) Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing (EMNLP 2024), pp. 13120–
13133. Association for Computational Linguistics (2024). doi:10.18653/v1/2024.
emnlp-main.728
22. Shorinwa, O., Mei, Z., Lidard, J., Ren, A.Z., Majumdar, A.: A survey on un-
certainty quantification of large language models: taxonomy, open research chal-
lenges, and future directions. ACM Comput. Surv. 58(3), Article 63 (2025).
doi:10.1145/3744238
23. Geng, J., Cai, F., Wang, Y., Koeppl, H., Nakov, P., Gurevych, I.: A survey of con-
fidence estimation and calibration in large language models. In: Duh, K., Gomez,
H., Bethard, S. (eds.) Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pp. 6577–6595. Association for Computational
Linguistics (2024). doi:10.18653/v1/2024.naacl-long.366
24. Karaca, Y., Moonis, M.: Shannon entropy-based complexity quantification of non-
linear stochastic process: diagnostic and predictive spatiotemporal uncertainty of
multiple sclerosis subgroups. In: Karaca, Y., Baleanu, D., Zhang, Y.-D., Ger-
vasi, O., Moonis, M. (eds.) Multi-Chaos, Fractal and Multi-Fractional Artificial

## Page 23

Contribution Title
23
Intelligence of Different Complex Systems, pp. 231–245. Academic Press (2022).
doi:10.1016/B978-0-323-90032-4.00018-3
25. Ray, S.N., Chattopadhyay, S.: Analyzing surface air temperature and rainfall in
univariate framework, quantifying uncertainty through Shannon entropy and pre-
diction through artificial neural network. Earth Sci. Inform. 14(1), 485–503 (2021).
doi:10.1007/s12145-020-00555-5
26. Wen, B., Xu, C., Han, B., Wolfe, R., Wang, L.L., Howe, B.: From human to
model overconfidence: evaluating confidence dynamics in large language mod-
els. In: NeurIPS 2024 Workshop on Behavioral Machine Learning (2024). https:
//openreview.net/forum?id=y9UdO5cmHs
27. Xu, C., Wen, B., Han, B., Wolfe, R., Wang, L.L., Howe, B.: Do language models
mirror human confidence? Exploring psychological insights to address overcon-
fidence in LLMs. In: Che, W., Nabende, J., Shutova, E., Pilehvar, M.T. (eds.)
Findings of the Association for Computational Linguistics: ACL 2025, pp. 25655–
25672. Association for Computational Linguistics (2025). doi:10.18653/v1/2025.
findings-acl.1316
28. Xie, J., Chen, A.S., Lee, Y., Mitchell, E., Finn, C.: Calibrating language models
with adaptive temperature scaling. In: Al-Onaizan, Y., Bansal, M., Chen, Y.-N.
(eds.) Proceedings of the 2024 Conference on Empirical Methods in Natural Lan-
guage Processing (EMNLP 2024), pp. 18128–18138. Association for Computational
Linguistics (2024). doi:10.18653/v1/2024.emnlp-main.1007
29. Kruse, M., Afshar, M., Khatwani, S., Mayampurath, A., Chen, G., Gao, Y.: Simple
yet effective: an information-theoretic approach to multi-LLM uncertainty quan-
tification. In: Christodoulopoulos, C., Chakraborty, T., Rose, C., Peng, V. (eds.)
Proceedings of the 2025 Conference on Empirical Methods in Natural Language
Processing (EMNLP 2025), pp. 30493–30504. Association for Computational Lin-
guistics (2025). doi:10.18653/v1/2025.emnlp-main.1551
30. Rosser, J., Redondo Garc´ıa, J.L., Penha, G., Palla, K., Bouchard, H.: Stream: scal-
ing up mechanistic interpretability to long context in LLMs via sparse attention.
arXiv preprint arXiv:2510.19875 (2026). https://arxiv.org/abs/2510.19875
31. Ranaldi, L.: Survey on the role of mechanistic interpretability in generative AI.
Big Data Cogn. Comput. 9(8), Article 193 (2025). doi:10.3390/bdcc9080193
32. Ma, X., Wang, J., Jiang, Y., Monazam Erfani, S., Liu, T., Bailey, J.: Cognitive
mirrors: exploring the diverse functional roles of attention heads in LLM reasoning.
arXiv preprint arXiv:2512.10978 (2025). https://arxiv.org/abs/2512.10978
33. Ling, C., Zhao, X., Lu, J., Deng, C., Zheng, C., Wang, J., Chowdhury, T., Li,
Y., Cui, H., Zhang, X., Zhao, T., Panalkar, A., Mehta, D., Pasquali, S., Cheng,
W., Wang, H., Liu, Y., Chen, Z., Chen, H., White, C., Gu, Q., Pei, J., Yang,
C., Zhao, L.: Domain specialization as the key to make large language models
disruptive: a comprehensive survey. ACM Comput. Surv. 58(3), Article 79 (2025).
doi:10.1145/3764579
34. Chen, Z., Chen, J., Gaidhani, M., Singh, A., Sra, M.: XplainLLM: a QA explanation
dataset for understanding LLM decision-making. In: OpenReview (2024). https:
//openreview.net/forum?id=Ba5KGabRe8
35. Artzy, A.B., Schwartz, R.: Attend first, consolidate later: on the importance of
attention in different LLM layers. In: Belinkov, Y., Kim, N., Jumelet, J., Mohebbi,
H., Mueller, A., Chen, H. (eds.) Proceedings of the 7th BlackboxNLP Workshop:
Analyzing and Interpreting Neural Networks for NLP, pp. 177–184. Association for
Computational Linguistics (2024). doi:10.18653/v1/2024.blackboxnlp-1.10

## Page 24

24
A. Adeseye et al
36. Faye, G., Ouerdane, W., Gadek, G., Gahbiche, S., Gatepaille, S.: A novel hybrid
approach for text encoding: cognitive attention to syntax model to detect online
misinformation. Data Knowl. Eng. 148, 102230 (2023). doi:10.1016/j.datak.2023.
102230
37. Pavlovic, M., Heinis, T., Tauheed, F., Karras, P., Ailamaki, A.: TRANSFORM-
ERS: robust spatial joins on non-uniform data distributions. In: 2016 IEEE 32nd
International Conference on Data Engineering (ICDE), pp. 673–684. IEEE (2016).
doi:10.1109/ICDE.2016.7498280
38. Lin, A., Li, J., Xiang, Y., Bian, W., Prasad, M.: Normal transformer: extracting
surface geometry from LiDAR points enhanced by visual semantics. IEEE Trans.
Intell. Veh. 9(10), 6172–6182 (2024). doi:10.1109/TIV.2024.3363174
39. Kharyuk, P., Matveev, S., Oseledets, I.: Exploring specialization and sensitivity of
convolutional neural networks in the context of simultaneous image augmentations.
arXiv preprint arXiv:2503.03283 (2026). https://arxiv.org/abs/2503.03283
40. Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E., Schutze, H.,
Goldberg, Y.: Measuring and improving consistency in pretrained language models.
Trans. Assoc. Comput. Linguist. 9, 1012–1031 (2021). doi:10.1162/tacl a 00410
41. Braverman, M., Chen, X., Kakade, S., Narasimhan, K., Zhang, C., Zhang, Y.:
Calibration, entropy rates, and memory in language models. In: Daum´e III, H.,
Singh, A. (eds.) Proceedings of the 37th International Conference on Machine
Learning (ICML 2020), vol. 119, pp. 1089–1099. PMLR (2020).
42. Alhindi, A., Al-Ahmadi, S., Ismail, M.M.B.: Advancements and challenges in
privacy-preserving split learning: Experimental findings and future directions. Int.
J. Inf. Secur. 24(3), 125 (2025).
43. Murala, D.K., Prasada Rao, K.V., Vuyyuru, V.A., Assefa, B.G.: A service-oriented
microservice framework for differential privacy-based protection in industrial IoT
smart applications. Sci. Rep. 15(1), 29230 (2025).
44. Huang, P.-Y., Chang, X., Hauptmann, A.: Multi-head attention with diversity for
learning grounded multilingual multimodal representations. In: Inui, K., Jiang, J.,
Ng, V., Wan, X. (eds.) Proceedings of the 2019 Conference on Empirical Methods
in Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pp. 1461–1467. Association for
Computational Linguistics, Hong Kong (2019).
45. Bakhtyari, N., Rejeb Bouzgarrou, A., Claramunt, C., Rejeb, H.: A disper-
sion index for the analysis of the distribution of activities in the Tunisian
coastal city of Nabeul. Geomatics 2(2), 161–180 (2022). https://doi.org/10.3390/
geomatics2020010
46. Kang, R., Zhang, Q., Zeng, Z., Zio, E., Li, X.: Measuring reliability under epistemic
uncertainty: Review on non-probabilistic reliability metrics. Chinese Journal of
Aeronautics 29(3), 571–579 (2016). https://doi.org/10.1016/j.cja.2016.04.004
47. Zhang, C., Lv, J., Cao, J., Sheng, J., Song, D., Zhang, T.: Unravelling the semantic
mysteries of transformers layer by layer. The Computer Journal 68(9), 1237–1251
(2025). doi:10.1093/comjnl/bxaf034
