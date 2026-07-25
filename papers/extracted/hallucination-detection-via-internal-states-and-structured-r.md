---
source_pdf: papers/HALLUCINATION DETECTION VIA INTERNAL STATES AND STRUCTURED REASONING.pdf
slug: hallucination-detection-via-internal-states-and-structured-r
pages: 5
extracted_on: 2026-07-22
---

# HALLUCINATION DETECTION VIA INTERNAL STATES AND STRUCTURED REASONING

## Page 1

HALLUCINATION DETECTION VIA INTERNAL STATES AND STRUCTURED REASONING
CONSISTENCY IN LARGE LANGUAGE MODELS
Yusheng Song1, Lirong Qiu1, Xi Zhang1, Zhihao Tang1,†
1 Beijing University of Posts and Telecommunications, Beijing, China
{songys, qiulirong, zhangx, innerone}@bupt.edu.cn
ABSTRACT
The detection of sophisticated hallucinations in Large Language
Models (LLMs) is hampered by a “Detection Dilemma”: methods
probing internal states (Internal State Probing) excel at identifying
factual inconsistencies but fail on logical fallacies, while those ver-
ifying externalized reasoning (Chain-of-Thought Verification) show
the opposite behavior. This schism creates a task-dependent blind
spot: Chain-of-Thought Verification fails on fact-intensive tasks like
open-domain QA where reasoning is ungrounded, while Internal
State Probing is ineffective on logic-intensive tasks like mathemati-
cal reasoning where models are confidently wrong. We resolve this
with a unified framework that bridges this critical gap. However,
unification is hindered by two fundamental challenges: the Signal
Scarcity Barrier, as coarse symbolic reasoning chains lack signals
directly comparable to fine-grained internal states, and the Repre-
sentational Alignment Barrier, a deep-seated mismatch between
their underlying semantic spaces. To overcome these, we introduce
a multi-path reasoning mechanism to obtain more comparable, fine-
grained signals, and a segment-aware temporalized cross-attention
module to adaptively fuse these now-aligned representations, pin-
pointing subtle dissonances. Extensive experiments on three diverse
benchmarks and two leading LLMs demonstrate that our framework
consistently and significantly outperforms strong baselines.
Our
code is available: https://github.com/peach918/HalluDet.
Index Terms— Natural language processing, Large language
models, Generative AI, Attention mechanisms, Machine learning
1. INTRODUCTION
Large Language Models (LLMs) are revolutionizing information in-
teraction, yet a propensity to “hallucinate”—generating plausible yet
false content—critically undermines the technology’s transformative
potential [1]. Hallucination presents a fundamental flaw that chal-
lenges LLM reliability [2], especially in high-stakes domains like
healthcare, finance, and law, where errors can lead to catastrophic
outcomes [3], [4]. The resulting application bottleneck erodes sys-
temic trust and prevents widespread adoption. Consequently, hallu-
cination detection has become a cornerstone challenge for ensuring
safe and trustworthy LLM applications.
Current efforts in hallucination detection are largely divided into
two isolated paradigms. First, the ‘neuroscientist’s path’ of Internal
State Probing (ISP) [5], [6] examines sub-symbolic signals within
the model—such as neural activation patterns [7], token generation
probabilities [8], or semantic entropy [9]—to find internal incon-
sistencies. Second, the ‘psychologist’s path’ of Chain-of-Thought
Verification (CoTV) [10], [11] analyzes the logical coherence of the
†Zhihao Tang is the corresponding author.
Q&A with LLM
CoT Answer
Q&A with LLM
Open-domain QA
Math Problem
ISP
Hallucination
Non-Hallu
CoTV
Unified
CoTV Hallucination detection
ISP Hallucination detection
Unified Hallucination detection(Ours)
Open-domain QA
Math Problem
Open-domain QA
Math Problem
Hallucination
Non-Hallu
Hallucination
Non-Hallu
(a) A visual comparison of the methods’
effectiveness and performance.
Open-domain QA Math Problem
60
70
80
90
AUROC (%)
ISP
CoTV
Unified(ours)
(b) Quantitative analysis of AU-
ROC performance.
Fig. 1: Effectiveness of detection methods for hallucination types.
model’s externalized, symbolic reasoning traces, often using self-
verification protocols to detect contradictions [10], [12]. However,
these two paradigms have evolved not in concert, but largely in
isolation. This “binary schism” in research is no accident; it reflects
the long-standing methodological divide in artificial intelligence
between sub-symbolic (connectionist) and symbolic (classicist) ap-
proaches [13]. This fracture creates a blind spot for detecting the
most dangerous and subtle hallucinations.
This schism creates the Detection Dilemma: a critical blind spot
where each paradigm fails in a complementary manner. As illus-
trated in Fig. 1, the failures are task-dependent. ISP methods, while
effective at gauging a model’s statistical certainty, are blind to logical
fallacies. They are thus ineffective in domains like mathematical rea-
soning, where a model can be highly confident in a logically flawed
answer [14]. Conversely, CoTV methods excel at verifying the in-
ternal coherence of a reasoning chain but cannot ground it in factual
reality. They consequently fail in open-domain QA, where mod-
els build logical arguments on a factually incorrect premise, yield-
ing self-consistent fabrications [15]. The essence of the Detection
Dilemma is this decoupling of statistical confidence from factual
grounding. Consequently, the most insidious hallucinations—those
that are both statistically confident and logically coherent, yet factu-
ally baseless—evade detection by either method alone.
To resolve the Detection Dilemma, ISP and CoTV must be uni-
fied, yet this path is blocked by two fundamental technical chal-
lenges. The first is the Signal Scarcity Barrier. CoTV typically
depends on a single reasoning path, which often appears logically
self-consistent and thus fails to anchor sub-symbolic anomalies to
concrete logical flaws. Consequently, anomalies detected by ISP
cannot be validated against explicit reasoning, while CoTV evidence
remains sparse. This lack of cross-paradigm indicators creates a se-
mantic gap, yielding a scarcity of reliable hallucination signals. The
second is the Representational Alignment Barrier. Even when
arXiv:2510.11529v2  [cs.CL]  8 Jan 2026

## Page 2

In France they all speak French. In 
England they all speak English. In Ireland 
they all speak what?
Multi-Path Signal Generation
Reasoning Granularity Alignment
Cross-Modal Fusion via Gated Cross-Attention
 In Ireland, they all speak Irish.
Question Analysis:The question is asking 
about ...... Synthesis:While Irish is the 
official language, English is the primary 
language spoken by the majority of the 
population.
Do people in Ireland speak Irish?
User
Semantic Trajectory List
Question Analysis: The question 
is asking about the language 
spoken in Ireland. 
Answer
CoT Answer
Reverse Query
LLM
Internal States Extraction
LLM
Cross-
Attention 
Classifier
INPUT
MULTI-HEAD ATTEN
ADD & NORM
FFN
OUTPUT
Inner 
Representation
Embedding of Query
Embedding of Reverse Query
...
Embedding of Semantic 
Trajectory List
Embedding of Answer
Training
The answer is ...
Sorry, I can not 
answer this question.
Non-Hallucination
Hallucination
LLM
LLM
Synthesis:While Irish is the 
official language, English is the 
primary language spoken by the 
majority of the population.
Seg ID 1
Seg ID 2
Seg ID 3
Seg ID K
...
It is a misconception that 
everyone in Ireland speaks the 
same language.
Fact Verification:Ireland has 
two official languages: Irish is 
the native language, while 
English is widely used.
Unified Consistency Verification Module
LLM
LLM
Fig. 2: Overview of the proposed hallucination detection framework.
signals are encoded into a shared embedding format, their underly-
ing semantic spaces remain heterogeneous. Embeddings of inter-
nal states capture latent statistical patterns, whereas embeddings of
reasoning traces capture compositional logic. A direct vector com-
parison is therefore unreliable, confounded by a severe mismatch
in both semantics and granularity (e.g., a fine-grained neural signal
versus a coarse-grained reasoning step). Overcoming this alignment
challenge is a critical prerequisite for successful unification.
A novel framework is introduced to resolve the Detection
Dilemma by enforcing consistency between a model’s internal
states and its externalized reasoning. This is operationalized through
two technical innovations. First, to overcome the Signal Scarcity
Barrier, a multi-path reasoning mechanism is employed to delib-
erately generate a diverse signal portfolio from both direct answers
and auxiliary Chain-of-Thought (CoT). The CoT is then decom-
posed into a structured Semantic Trajectory List. This critical step
transforms the coarse symbolic trace into a fine-grained sequence,
creating an explicit bridge that makes symbolic logic directly com-
parable to sub-symbolic neural states.
Second, to overcome the
Representational Alignment Barrier, a segment-aware temporal-
ized cross-Attention module is proposed. This component unifies
the heterogeneous embeddings from questions, answers, and the
now-structured CoT trajectories into a coherent representational
space. By adaptively aligning these modalities, our module effec-
tively detects the subtle semantic dissonances that are the hallmarks
of sophisticated hallucinations. Extensive experiments on three pub-
lic benchmarks validate our framework’s effectiveness, consistently
outperforming strong baselines.
Our main contributions are summarized as follows:
• We formally identify the Detection Dilemma in current re-
search and propose the first unified framework to resolve it by
bridging sub-symbolic and symbolic model representations.
• We introduce two technical innovations: a multi-path reason-
ing mechanism to address signal scarcity and a temporalized
cross-attention to resolve representational misalignment.
• We demonstrate state-of-the-art performance on three diverse
benchmarks, establishing a new standard for reliable halluci-
nation detection.
2. METHODOLOGY
To resolve the ‘Detection Dilemma’, we introduce a framework
that integrates an LLM’s internal sub-symbolic states with its ex-
ternalized symbolic reasoning (Fig. 2).
Our design overcomes
signal scarcity and representational misalignment via two innova-
tions: a multi-path process to generate diverse signals, and a unified
verification module to fuse them for discrepancy analysis. These
components are detailed below.
2.1. Multi-Path Signal Generation for Comprehensive Diagnos-
tics
To overcome the signal scarcity barrier, our framework generates
signals from three complementary reasoning paths for any input
query Q, constructing a rich diagnostic landscape. This strategy
performs cognitive triangulation, forcing the model to approach
a problem from multiple angles to amplify latent inconsistencies
indicative of hallucinations. The three paths are defined as follows:
1. Direct Answer Path: The LLM is prompted to generate a
direct answer, Adir, without explicit intermediate reasoning.
This path captures the model’s spontaneous, unconditioned
output, providing a baseline assessment of its immediate fac-
tual recall and statistical confidence.
2. Reasoning-Augmented Path: The query Q is re-prompted
with a Chain-of-Thought [16] (CoT) instruction to elicit Acot,
a detailed response externalizing the model’s step-by-step
symbolic reasoning. This path renders the model’s logical
trajectory transparent and amenable to verification.
3. Reverse-Inference Path: The direct answer Adir is supplied
back to the LLM with the objective of inferring a plausible
original query, Qrev, that would logically lead to it. This path
functions as a crucial semantic consistency check, probing
whether the generated answer is sufficiently grounded to en-
tail a question that aligns with the original query’s intent.
This tri-path generation strategy yields multi-perspective paired
data (Q-Adir, Q-Acot, Adir-Qrev), which forms the foundation for our
subsequent cross-modal consistency analysis.
For supervised training, high-quality hallucination labels are
generated via a LLM-as-a-Judge protocol, which employs two state-
of-the-art LLMs (GPT-4.1 and Gemini-2.5 Pro) to independently
verify the target model’s query–answer pairs. Each pair is assigned
a binary label (0 for non-hallucination, 1 for hallucination). Pairs re-
ceiving concordant labels from both judges are incorporated directly
into our dataset. Domain experts manually resolve disagreements to
efficiently produce a large-scale, highly accurate labeled dataset.
2.2. Unified Consistency Verification Module
The core technical engine of our framework is a unified verification
module engineered to resolve the representational alignment barrier.

## Page 3

1
5
9
13 17 21 25 2932
50
55
60
65
70
75
80
85
90
AUROC [%]
Layer
(a) Effect of Layer
0.2
0.4
0.6
0.8
65
70
75
80
85
90
AUROC [%]
Temperature
(b) Effect of Temperature
50100
200
300
65
70
75
80
85
90
AUROC [%]
Sequence Length
(c) Effect of Sequence Length
24 8
16
32
65
70
75
80
85
90
AUROC [%]
Attention Heads
(d) Effect of Attention Heads
Fig. 3: Hyperparameter analysis for key parameters and their impact on AUROC.
This module addresses the complex challenge of comparing and in-
tegrating heterogeneous signals—namely, the sub-symbolic hidden
states from the LLM’s neural pathways and the symbolic, structured
text from the reasoning-augmented path. Its architecture follows a
two-stage process: first, aligning the granularity of the symbolic rea-
soning trace with internal state representations, and second, merging
them to detect semantic and logical dissonances.
2.2.1. Reasoning Granularity Alignment via Temporal Modeling
To align the variable-length symbolic CoT response Acot with fixed-
dimensional neural representations, a semantic trajectory decompo-
sition is first performed. Specifically, Acot is segmented into a se-
quence of minimal, coherent semantic units u1, u2, . . . , um, termed
the Semantic Trajectory List (STL). This segmentation follows lin-
guistic cues like logical connectors (e.g., “therefore”, “because”),
causal transitions, and fact-introduction points to maintain the in-
tegrity of each reasoning step. The resulting STL is then subjected
to temporal embedding, enabling structured and fine-grained model-
ing of the reasoning process in a neural representation space.
The resulting STL offers a structured representation of the rea-
soning process, defined as a sequence of embeddings:
T = [e1, e2, . . . , em],
(1)
where ei = Enc(ui) is the embedding of the i-th reasoning unit,
obtained from the same encoder as the answer-generating LLM. This
clause-level representation captures both the temporal progression of
the logic and the fine-grained semantic shifts between steps.
To distill the sequential information into a compact representa-
tion, temporal modeling is employed. Specifically, a learnable clas-
sification token [CLS] is prepended to the sequence of trajectory
embeddings and processed with a Transformer encoder [17]. This
architecture is chosen for its ability to capture long-range dependen-
cies, essential for reasoning chains where a conclusion depends on
a distant premise. The final aggregated representation of the CoT
path, hCoT, is extracted from the output state of the token:
hCoT = Enc([[CLS]; e1, . . . , eT ])[CLS].
(2)
This procedure performs semantic compression, creating a
holistic vector that encapsulates the entire reasoning trajectory while
aligning its granularity with other internal state representations.
2.2.2. Cross-Modal Fusion via Gated Cross-Attention
With all signals transformed into a shared representational format,
the final stage involves their integration and analysis to detect incon-
sistencies. This fusion process follows a hierarchical verification:
first ensuring consistency within the sub-symbolic domain, then per-
forming a cross-modal check against the symbolic reasoning trace.
Internal State Extraction and Contextualization. For the non-
CoT paths (Q, Adir, Qrev), their vector representations are obtained in
two ways. For Adir and Qrev, the corresponding hidden states are di-
rectly extracted from the LLM during generation. For Q, as an input
rather than a generated output, it is fed into the same LLM, with its
embedding layer used to produce EQ. To preserve the origin and role
of each representation, a unique Segment ID is assigned to each em-
bedding (e.g., one for queries, another for answers). Subsequently,
EQ, EAdir, and EQrev are concatenated into a sequence Xmain, which
is passed through a Multi-Head self-Attention [17] (MHA) block to
perform an intra-modal consistency check, yielding Hmain that cap-
tures relationships and discrepancies among these signals.
Hmain = MHA(LayerNorm(Xmain + Eseg)).
(3)
Adaptive Reasoning Gate. To dynamically regulate the influence
of the symbolic reasoning path, a gating mechanism is introduced.
A scalar gate g ∈R is computed from the contextualized internal
states Hmain and applied to modulate the CoT representation hCoT.
This enables the model to down-weight the reasoning trace if internal
signals deem it unreliable or irrelevant for a given instance.
g = σ(FFN(Hmain)),
ˆhCoT = g · hCoT.
(4)
Cross-Attention for Discrepancy Detection. The core of our verifi-
cation is a final inter-modal consistency check, implemented through
a cross-attention [18] module.
The contextualized internal states
Hmain serve as a set of queries to probe the gated symbolic reason-
ing representation ˆhCoT, which provides the key-value context. The
output, Z, is a fused representation where dissonances between the
model’s sub-symbolic “knowledge” and symbolic “explanation” are
highlighted by the attention mechanism.
Z = CrossAttn(Hmain, ˆhCoT).
(5)
This fused representation Z is then passed through a final MLP clas-
sifier to produce the logits l ∈R2 for hallucination prediction.
Optimization. Due to the natural class imbalance between hallu-
cinated and factual statements, the model is optimized using Focal
Loss [19] (LFL), which prioritizes hard-to-classify examples:
LFL = −αt(1 −pt)γ log(pt),
(6)
where pt is the model’s estimated probability for the ground-truth
class, γ is a focusing parameter, and αt is a weighting factor to
balance class importance. This cross-attention-based fusion archi-
tecture enables unified, end-to-end modeling that captures subtle yet
critical hallucination signals by identifying patterns of disagreement
across different modalities of the model’s own cognitive processes.
3. EXPERIMENTS
3.1. Experimental Setup
To ensure methodological rigor, we evaluate on two distinct LLMs,
LLaMA2-7B-Chat [23] and Qwen2.5-7B [24], to demonstrate gen-
eralizability. Our testbed embodies the “Detection Dilemma” using
three benchmarks: fact-intensive TruthfulQA [25], logic-intensive
GSM8K [26], and TriviaQA [27]. We compare against state-of-the-
art ISP (HaloScope [6], SAPLMA [20]) and CoTV (V-STaR [22])

## Page 4

Table 1: Main hallucination detection results. Best baseline results
are underlined. Gains of our method are highlighted in green.
LLM
Method
TruthfulQA
TriviaQA
GSM8K
Qwen2.5-7B
SAPLMA [20]
59.66 ± 1.69
62.36 ± 1.38
59.72 ± 1.91
selfcheckgpt [21]
55.08 ± 1.15
74.65 ± 0.92
67.98 ± 1.28
semantic entropy [9]
64.72 ± 1.26
75.68 ± 0.88
58.36 ± 1.46
V-STaR [22]
63.91 ± 0.93
71.09 ± 1.15
76.55 ± 1.21
HaloScope [6]
79.31 ± 2.33
81.52 ± 2.08
70.36 ± 2.46
ours
84.03 ± 1.69
85.68 ± 1.71
79.15 ± 1.68
Gain vs. Best
+4.72
+4.16
+2.60
Llama2-7B-chat
SAPLMA [20]
57.41 ± 1.71
60.32 ± 1.83
58.64 ± 2.02
selfcheckgpt [21]
52.95 ± 1.20
73.22 ± 1.01
62.36 ± 1.35
semantic entropy [9]
62.17 ± 1.32
73.65 ± 0.77
57.46 ± 1.53
V-STaR [22]
61.28 ± 1.12
68.83 ± 1.33
74.38 ± 1.25
HaloScope [6]
78.64 ± 2.25
77.40 ± 1.98
65.79 ± 2.31
ours
82.42 ± 1.63
79.46 ± 1.87
76.83 ± 2.06
Gain vs. Best
+3.78
+2.06
+2.45
baselines. Performance is measured by AUROC, with all decoding
at a fixed temperature of 0.8 and a maximum length of 300 tokens.
3.2. Quantitative Performance Comparison
As presented in Table 1, our unified framework consistently and sig-
nificantly outperforms all baselines, providing robust empirical evi-
dence for its superiority in resolving the “Detection Dilemma”. This
dilemma is empirically manifested in the specialized performance
of prior methods: the CoTV-based V-STaR [22] excels on logic-
intensive tasks (GSM8K: 76.55%) but fails on fact-intensive ones
(TruthfulQA: 63.91%), while the ISP-based HaloScope [6] exhibits
the opposite trade-off (GSM8K: 70.36% vs. TruthfulQA: 79.31%).
Our framework breaks this trade-off, achieving state-of-the-art per-
formance on both TruthfulQA (84.03%) and GSM8K (79.15%) si-
multaneously. This balanced, high-level performance across funda-
mentally different tasks demonstrates that by successfully unifying
sub-symbolic and symbolic signals, our approach overcomes prior
weaknesses to achieve a more generalized detection capability.
3.3. In-depth Analysis
To analyze the framework’s performance, a series of analyses was
conducted to explain not just what it achieves, but why it is effective.
Component-wise Efficacy.
An ablation study (Fig. 5) reveals a
strong synergy between our primary innovations. Removing Inter-
nal States (w/o Internal), the CoT verification path (w/o CoT), or the
Reverse Inference Path (w/o Reverse) degrades performance, and
the full model’s improvement is non-linear. On TruthfulQA, our
full model achieves a 4.12-point AUROC improvement over the best
component. This suggests our cross-attention fusion deeply inte-
grates the signals—CoT for transparent reasoning, Internal States for
statistical patterns, and Reverse Inference for semantic consistency.
This synergy is key to resolving the Detection Dilemma.
t-SNE of only internal states
Non-Hallucination
Hallucination
t-SNE of unified representations
Non-Hallucination
Hallucination
Fig. 4: Comparison of t-SNE projections. The visualization distin-
guishes hallucination (red) from non-hallucination (blue) samples.
TruthfulQA
TriviaQA
GSM8K
60
65
70
75
80
85
90
AUROC (%)
74.85
72.55
65.20
68.45
67.45
71.05
78.30
76.10
73.95
82.42
79.46
76.83
w/o CoT
w/o Internal
w/o Reverse
Full Model
Fig. 5: Ablation Study:Internal States, CoT & Reverse Inference.
Model Characteristics and Interpretability.
Hyperparameter
analysis (Fig. 3) provides further insight into the model’s operational
logic. Optimal performance is achieved using late-stage represen-
tations (24th layer), confirming that hallucination detection relies
on abstract semantic features. The framework’s accuracy is highest
for text generated at a higher temperature (T = 0.8), indicating
it is most effective in the creative (and thus higher-risk) scenarios
where it is most needed. To provide intuitive visual evidence of
the framework’s mechanics, qualitative analyses were performed.
A t-SNE [28] projection of the feature space (Fig. 4) shows that
our unified representations achieve significantly better separation
between hallucinated (red) and non-hallucinated (blue) samples
compared to using only internal states, demonstrating superior dis-
criminative power.
Furthermore, visualizing the cross-attention
weights (Fig. 6) provides interpretability. In the given example, the
model correctly places high attention on tokens that create a seman-
tic dissonance between a false statement (“In Ireland they all speak
Irish”) and corrective facts in the CoT (“two official languages,”
“English spoken widely”). This confirms that the framework’s de-
cisions are grounded in identifiable semantic contradictions rather
than opaque correlations, enhancing trust in its predictions.
Ireland
has
two
official
languages
:
Irish
and
English
.
English
is
widely
spoken
.
CoT Answer Tokens
In
Ireland
,
they
all
speak
Irish
.
Hallucination Answer Tokens
0.2
0.4
0.6
0.8
Attention Weight
Fig. 6: Visualization of cross-attention weights between a halluci-
nated answer and the corresponding CoT trace.
4. CONCLUSION
In this work, we address the “Detection Dilemma” in LLM halluci-
nation, a vulnerability from the schism between Internal State Prob-
ing and Chain-of-Thought Verification, by introducing the first uni-
fied framework to bridge sub-symbolic and symbolic signals. Our
approach overcomes the Signal Scarcity Barrier with a multi-path
reasoning mechanism and the Representational Alignment Barrier
via a segment-aware temporalized cross-Attention module. Signifi-
cant performance gains across diverse benchmarks validate our the-
sis that this synergistic approach is essential for robust detection,
representing a critical step towards building trustworthy LLMs for
high-stakes applications.

## Page 5

5. REFERENCES
[1] Wayne Xin Zhao, Kun Zhou, Junyi Li, et al., “A survey of large
language models,” arXiv preprint arXiv:2303.18223, vol. 1,
no. 2, 2023.
[2] Adam Tauman Kalai, Ofir Nachum, Santosh S Vempala,
et al.,
“Why language models hallucinate,”
arXiv preprint
arXiv:2509.04664, 2025.
[3] Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin,
Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese,
Borja Balle, Atoosa Kasirzadeh, et al.,
“Ethical and so-
cial risks of harm from language models,”
arXiv preprint
arXiv:2112.04359, 2021.
[4] Lei Huang, Weijiang Yu, Weitao Ma, et al., “A survey on hallu-
cination in large language models: Principles, taxonomy, chal-
lenges, and open questions,” arXiv preprint arXiv:2311.05232,
2023.
[5] Xiaokang Zhang, Zijun Yao, Jing Zhang, et al., “Transferable
and efficient non-factual content detection via probe training
with offline consistency checking,” in Proceedings of the 62nd
Annual Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers), 2024, pp. 12348–12364.
[6] Xuefeng Du, Chaowei Xiao, and Sharon Li, “Haloscope: Har-
nessing unlabeled llm generations for hallucination detection,”
Advances in Neural Information Processing Systems, vol. 37,
pp. 102948–102972, 2024.
[7] Weihang Su, Changyue Wang, Qingyao Ai, et al.,
“Unsu-
pervised real-time hallucination detection based on the inter-
nal states of large language models,” in Findings of the As-
sociation for Computational Linguistics ACL 2024, 2024, pp.
14379–14391.
[8] Ernesto Quevedo, Jorge Yero Salazar, Rachel Koerner, et al.,
“Detecting hallucinations in large language model generation:
A token probability approach,” in World Congress in Com-
puter Science, Computer Engineering & Applied Computing.
Springer, 2024, pp. 154–173.
[9] Jiatong Han, Jannik Kossen, Muhammed Razzak, Lisa Schut,
Shreshth A Malik, and Yarin Gal, “Semantic entropy probes:
Robust and cheap hallucination detection in llms,” in ICML
2024 Workshop on Foundation Models in the Wild, 2024.
[10] Yixuan Weng, Minjun Zhu, Fei Xia, et al., “Large language
models are better reasoners with self-verification,”
in Find-
ings of the Association for Computational Linguistics: EMNLP
2023, Houda Bouamor, Juan Pino, and Kalika Bali, Eds., Sin-
gapore, Dec. 2023, pp. 2550–2575, Association for Computa-
tional Linguistics.
[11] Tianci Xue, Ziqi Wang, Zhenhailong Wang, Chi Han, Pengfei
Yu, and Heng Ji, “Rcot: Detecting and rectifying factual in-
consistency in reasoning by reversing chain-of-thought,” arXiv
preprint arXiv:2305.11499, 2023.
[12] Yifei Li, Zeqi Lin, Shizhuo Zhang, et al., “Making language
models better reasoners with step-aware verifier,” in Proceed-
ings of the 61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), 2023, pp. 5315–
5333.
[13] Ron Sun, “Artificial intelligence: Connectionist and symbolic
approaches,” 2001.
[14] Mohammad Beigi, Ying Shen, Runing Yang, et al.,
“Inter-
nalinspector i2: Robust confidence estimation in llms through
internal states,” in Findings of the Association for Computa-
tional Linguistics: EMNLP 2024, 2024, pp. 12847–12865.
[15] Ruixin Hong, Hongming Zhang, Xinyu Pang, et al., “A closer
look at the self-verification abilities of large language models
in logical reasoning,” in Proceedings of the 2024 Conference
of the North American Chapter of the Association for Compu-
tational Linguistics: Human Language Technologies (Volume
1: Long Papers), 2024, pp. 900–925.
[16] Jason Wei, Xuezhi Wang, Dale Schuurmans, et al., “Chain-of-
thought prompting elicits reasoning in large language models,”
Advances in neural information processing systems, vol. 35,
pp. 24824–24837, 2022.
[17] Ashish Vaswani, Noam Shazeer, Niki Parmar, et al., “Attention
is all you need,” Advances in neural information processing
systems, vol. 30, 2017.
[18] Chun-Fu Richard Chen, Quanfu Fan, and Rameswar Panda,
“Crossvit: Cross-attention multi-scale vision transformer for
image classification,” in Proceedings of the IEEE/CVF inter-
national conference on computer vision, 2021, pp. 357–366.
[19] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and
Piotr Doll´ar, “Focal loss for dense object detection,” in Pro-
ceedings of the IEEE international conference on computer vi-
sion, 2017, pp. 2980–2988.
[20] Amos Azaria and Tom Mitchell, “The internal state of an llm
knows when it’s lying,” in Findings of the Association for Com-
putational Linguistics: EMNLP 2023, 2023, pp. 967–976.
[21] Potsawee Manakul, Adian Liusie, and Mark Gales,
“Self-
checkgpt: Zero-resource black-box hallucination detection for
generative large language models,” in The 2023 Conference on
Empirical Methods in Natural Language Processing.
[22] Arian Hosseini, Xingdi Yuan, Nikolay Malkin, et al., “V-star:
Training verifiers for self-taught reasoners,” in Conference on
Language Modeling, 2024.
[23] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Am-
jad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya
Batra, Prajjwal Bhargava, Shruti Bhosale, et al.,
“Llama 2:
Open foundation and fine-tuned chat models,” arXiv preprint
arXiv:2307.09288, 2023.
[24] A Yang Qwen, Baosong Yang, B Zhang, B Hui, B Zheng,
B Yu, Chengpeng Li, D Liu, F Huang, H Wei, et al., “Qwen2.5
technical report,” arXiv preprint, 2024.
[25] Stephanie Lin, Jacob Hilton, and Owain Evans, “Truthfulqa:
Measuring how models mimic human falsehoods,” in Proceed-
ings of the 60th Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), 2022, pp. 3214–
3252.
[26] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al.,
“Training verifiers to solve math word problems,”
arXiv
preprint arXiv:2110.14168, 2021.
[27] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettle-
moyer, “Triviaqa: A large scale distantly supervised challenge
dataset for reading comprehension,” in Proceedings of the 55th
Annual Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers), 2017, pp. 1601–1611.
[28] Laurens van der Maaten and Geoffrey Hinton, “Visualizing
data using t-sne,” Journal of machine learning research, vol.
9, no. Nov, pp. 2579–2605, 2008.
