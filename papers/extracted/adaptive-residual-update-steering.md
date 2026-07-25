---
source_pdf: C:/Users/omris/TAU/hallucination_detection/papers/Adaptive_Residual_Update_Steering.pdf
slug: adaptive-residual-update-steering
pages: 20
extracted_on: 2026-07-15
---

# Adaptive_Residual_Update_Steering

## Page 1

Adaptive Residual-Update Steering for Low-Overhead Hallucination
Mitigation in Large Vision Language Models
Zhengtao Zou 1 Ya Gao 1 Jiarui Guan 1 Bin Li 2 * Pekka Marttinen 1 *
Abstract
Large Vision-Language Models (LVLMs) typi-
cally process visual inputs as a prefix to the lan-
guage decoder. As the model autoregressively
generates text, this initial visual information in-
evitably undergoes “dilution”, leading the model
to over-rely on language priors and hallucinate
objects. Existing interventions attempt to correct
this by contrasting logits or iteratively refining
outputs, but they incur prohibitive latency costs.
We propose Residual-Update Directed DEcoding
Regulation (RUDDER), a framework that coun-
ters visual dilution by creating a persistent visual
anchor. We extract a robust evidence direction
(CARD) directly from the model’s prefill residual
updates, and inject it into the decoding process.
This injection is modulated by an adaptive gate,
the Beta Gate, which acts as a trust mecha-
nism and ensures the visual reminder is applied
only when necessary. Experiments on LLaVA-1.5
(7B/13B), Idefics2, InstructBLIP, and Qwen2.5-
VL demonstrate that RUDDER consistently miti-
gates hallucination (with greedy decoding, RUD-
DER reduces CHAIRS by an average of 24.4%
and CHAIRi by 23.6% relative) and scales effec-
tively across architectures, all while maintaining
96.0% throughput. The code is available at RUD-
DER.
1. Introduction
Large Vision-Language Models (LVLMs) usually encode
visual inputs as a prefix to the language decoder. Since
the model generates text autoregressively, this initial vi-
*Equal contribution
1Aalto University, Espoo, Finland
2Shenzhen
Institutes
of
Advanced
Technology,
Chinese
Academy
of
Sciences,
Shenzhen,
China.
Correspon-
dence to:
Bin Li <b.li2@siat.ac.cn>,
Pekka Marttinen
<pekka.marttinen@aalto.fi>.
Proceedings of the 43 rd International Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).
sual information inevitably undergoes “dilution” (Liu et al.,
2024b), resulting in the model over-relying on internal lan-
guage priors (Leng et al., 2023). This phenomenon is a
primary driver of object hallucination. As illustrated in
Figure 1 (Left), the model drifts from the visual context,
hallucinating a “cup” based on the linguistic association
with “surface,” despite no such object being present.
To mitigate this, existing Inference-Time Intervention (ITI)
methods usually employ Contrastive Decoding (Leng et al.,
2023) or Iterative Refinement (Zhang et al., 2025). How-
ever, as summarized in Figure 1(Right), these approaches
face a severe bottleneck: they typically require multiple for-
ward passes (extra FWD) or external classifiers, incurring
prohibitive latency costs. This undermines the efficiency
of the autoregressive loop, making real-world deployment
difficult.
To resolve this efficacy-efficiency trade-off, we propose a
direct solution: Instead of correcting the model after it drifts,
we structurally prevent the dilution of visual information.
Our key insight is that the visual signal is maximally pre-
served in the residual stream during the mandatory prefill
stage. By capturing this signal before it fades, we can cre-
ate a lightweight “visual anchor” without any additional
forward pass.
We introduce Residual-Update Directed DEcoding Regula-
tion (RUDDER). RUDDER extracts the visual evidence as
a Contextual Activation Residual Direction (CARD) vector
from the prefill stage. During decoding, RUDDER continu-
ously “reminds” the LLM of this evidence by injecting the
CARD vector into the hidden states. To ensure this injection
is context-aware (“Per-token Adaptive” feature in Figure 1),
we employ Beta Gate, an adaptive gate based on the
Beta Distribution. This gate functions as a trust mechanism:
it dynamically reinforces the injection when the generation
aligns with the visual anchor, and suppresses it when the
model is processing non-visual (syntactic) tokens.
Our contributions are:
1. We propose CARD, a method to extract a persistent vi-
sual anchor from prefill residual updates. CARD keeps
the model grounded in the visual context throughout
1
arXiv:2511.10292v2  [cs.CV]  19 May 2026

## Page 2

Adaptive Residual-Update Steering for Hallucination Mitigation
Vanilla:  
“…There are several books 
scattered across the room, 
along with a cup placed on a 
surface…” 👻
Non-steering
Steering-based
Ours: RUDDER
No Extra FWD
Per-token Adaptive
Answer-span Localized
No Offline Setup
Typical Overhead
❌
❌
❌
✅
❌
❌
❌
✅
✅
✅
✅
✅
High
Mid-High
Baseline⚡
≈
Prompt:  
“Please help me describe the image in 
detail.” 
RUDDER😀: 
“…There are multiple books 
visible on a table or shelf, 
suggesting that this living 
area may be used for 
studying or enjoying various 
forms of reading material…” 
Figure 1. (Left) An example where the vanilla LLaVA-1.5–7B (Liu et al., 2024a) hallucinates objects. Erroneous text is marked in red,
while RUDDER’s corrected, factual output is in blue. (Right) A comparison showing that unlike existing non-steering and steering-based
methods, RUDDER provides adaptive, low-overhead control without requiring extra forward passes.
the decoding process.
2. We design the Beta Gate, a training-free adaptive gating
mechanism based on the Beta Distribution. It ensures
that the visual reminder is applied precisely, balancing
grounding with fluency.
3. We demonstrate that RUDDER is general and works
with different base models. It scales to larger models,
such as LLaVA-1.5-13B, and generalizes to modern
architectures with distinct fusion strategies, such as
Qwen2.5-VL and InstructBLIP.
4. Our method introduces little overhead to the generation
process. RUDDER operates within a single forward
pass with negligible latency (< 4%), significantly out-
performing multi-pass methods.
Conflict of Interest Disclosure.
The authors declare no
financial conflicts of interest related to this work.
2. Related Work
Our research is situated at the intersection of inference-time
intervention (ITI) and probabilistic gating.
Inference-time intervention. ITI aims to guide a model’s
generative behavior without modifying its weights. We
group existing methods based on where they act on the
computation path.
Non-steering methods operate at the output logits or ad-
just the decoding strategy.
Many of these methods re-
calibrate final logits to improve visual grounding, but of-
ten at the cost of significant latency due to extra forward
passes. For instance, VCD (Leng et al., 2023) uses per-
turbed images to create a negative context, PAI (Liu et al.,
2024b) subtracts unconditional (text-only) logits, and MA-
RINE (Zhao et al., 2025) employs a classifier-free guidance
style. Recent advancements include HALC (Chen et al.,
2024), which introduces adaptive focal-contrast decoding,
and AGLA (An et al., 2025), which mitigates hallucinations
by assembling global and local attention features. Addi-
tionally, DeGF (Zhang et al., 2025) proposes self-correcting
decoding via generative feedback, though such iterative re-
finement typically incurs higher computational overhead.
Similarly, DoLa (Chuang et al., 2023) contrasts deep vs.
shallow logits to suppress generic text. More efficient al-
ternatives, such as constrained decoding (Hokamp & Liu,
2017) or post hoc editing (Manakul et al., 2023), are typi-
cally less adaptive.
Steering methods directly modify the hidden representations
to guide the generation trajectory. Most of these methods
also incur high computational costs on-the-fly. For example,
ASD (Su et al., 2025) steers away from a predefined halluci-
nation direction, and VISTA (Li et al., 2025) injects a signal
vector computed from activation differences. VTI (Liu et al.,
2025) attempts to mitigate this cost by shifting the computa-
tional burden to an offline precomputation step.
Bayesian and Probabilistic Gating. Our work is also in-
spired by Bayesian and probabilistic gating for uncertainty
modeling. This includes concepts from Evidential Deep
Learning (Sensoy et al., 2018), which frames outputs as
parameters of a Dirichlet distribution for uncertainty quan-
tification. Other relevant work explores stochastic gates.
For instance, (Yamada et al., 2020) use stochastic gates
based on a relaxation of the Bernoulli distribution for fea-
ture selection. More directly related to our method, Beta-
LSTM (Song et al., 2019) replaces standard sigmoid gates
with ones derived from a Beta distribution, validating the
use of Bayesian principles in gating mechanisms.
3. Our Method
In autoregressive LVLMs, visual information tends to fade
as the generation sequence lengthens, a phenomenon often
referred to as visual fading or attention decay (Liu et al.,
2

## Page 3

Adaptive Residual-Update Steering for Hallucination Mitigation
2024b). However, during the initial prefill stage, the fusion
of visual and textual information is at its peak. The residual
update from the self-attention sublayer, therefore, encodes
the net effect of the visual context on the representation of
each text token. We hypothesize that by aggregating these
updates over the image tokens and text prompt tokens in
the prefill span, we can obtain a robust, per-sample vector
that captures the direction of visual evidence for the specific
input. Our empirical analysis supports this: the extracted
CARD vector creates a systematic, image-conditioned rota-
tion away from a text-only (language prior) direction, and
this rotation aligns coherently with the downstream steering
mechanism. This confirms the aggregated updates provide a
meaningful directional signal rather than random noise (a de-
tailed visualization and quantification is in Appendix A.4).
To mitigate object hallucination without the high compu-
tational costs of existing steering methods, we present
Residual-Update Directed DEcoding Regulation (RUD-
DER). RUDDER is a low-overhead framework that adap-
tively steers LVLMs toward visually-grounded generation
by injecting a dynamically derived visual evidence vector.
Crucially, it delivers context-specific steering within a single
forward pass and introduces negligible latency. RUDDER
is training-free; its few deployment hyperparameters are
selected once on a small held-out validation set, rather than
learned from task-specific supervision.
3.1. Preliminaries
The decoder in a Transformer-based LVLM operates on a
residual stream, where each sublayer’s output (e.g., self-
attention of the decoder layer) is added back to its input.
This output, termed the residual updates ∆l, represents the
new information contributed at layer l. We leverage these
updates during the two-stage auto-regressive generation pro-
cess: 1. Prefill Stage: The model processes the prefill span,
comprising both image tokens and text prompt tokens, in a
single parallel forward pass to populate a Key-Value cache.
During this mandatory step, we extract the CARD vector
by aggregating the self-attention residual updates across all
tokens in the prefill span. 2. Decoding Stage: The model
generates the output sequentially, one token at a time. It’s
during this phase that we employ Beta Gate for adaptive
steering.
Scope of the intervention.
RUDDER acts in the hidden
residual stream rather than at the final logit distribution. Its
objective is therefore not to provide a formal guarantee on
the token-level likelihood, but to preserve and reassert an
input-specific visual evidence direction during decoding.
This design matches the failure mode we target: hallucina-
tion often arises when the generation trajectory drifts from
the visual prefix toward language priors, while the origi-
nal visual evidence is still present in the model’s internal
computation during prefill.
3.2. CARD Vector: A Zero-Cost Per-Sample Evidence
Direction
Motivation. LVLMs fuse visual and textual information
through self-attention. The residual update from the self-
attention sublayer, therefore, encodes the net effect of the
visual context on the representation of each text token. We
hypothesize that by aggregating these updates over the im-
age tokens and text prompt tokens in the prefill span, we
can obtain a robust, per-sample vector that captures the di-
rection of visual evidence for the specific input (Liu et al.,
2024a). Our empirical analysis supports this: the extracted
CARD vector creates a systematic, image-conditioned rota-
tion away from a text-only (language prior) direction, and
this rotation aligns coherently with the downstream steering
mechanism. This confirms the aggregated updates provide a
meaningful directional signal rather than random noise (a de-
tailed visualization and quantification is in Appendix A.4).
To identify the optimal layer for extracting the CARD vector,
we analyze internal dynamics of LLaVA-1.5–7B (Liu et al.,
2024a). We find that intervening in the late decoder layers
has the greatest potential to influence the model’s final out-
put. Full analysis is provided in Appendix B.1, Figure 9a,
9c.
Extraction. In a single standard prefill pass with the image
and text prompt, we place a lightweight, read-only hook at
the target decoder layer l and cache the self-attention output
for each token i in the prefill span Tpre, denoted Al
i. In a
pre-norm decoder, the residual update is simply the attention
output,
∆l
i = Al
i,
(1)
We then pool these updates and apply L2 normalization to
obtain a per-sample direction:
vCARD =
Pool
 {∆l
i}i∈Tpre

 Pool
 {∆l
i}i∈Tpre

2
,
where Pool(·) can be mean or ∥∆l
i∥-weighted mean.
(2)
Intuition. Although pooling aggregates strictly over the
prefill span, it naturally prioritizes semantic content. Since
self-attention updates are magnitude-heavy on informative
tokens and lighter on functional syntax, the weighted pool-
ing effectively filters out noise, distilling a robust, input-
specific evidence direction that summarizes the net visual
influence. This entire process occurs within the single prefill
pass and introduces negligible overhead, as no additional
forward pass or calibration is required.
3

## Page 4

Adaptive Residual-Update Steering for Hallucination Mitigation
3.3. Beta Gate: Adaptive Gating Based on the Beta
Distribution
Motivation: A Trust Mechanism.
Standard steering of-
ten applies a fixed vector, which can be harmful when the
visual context is irrelevant (e.g., on syntactic tokens). To
address this, we introduce an adaptive gate derived from a
probabilistic perspective. Let hl,t be the hidden state for
generating token t at the target layer l. We measure its
alignment with the visual evidence via cosine similarity:
st = cos(hl,t, vCARD).
We use the function of a Beta-Bernoulli conjugate posterior
to design the gate. We view st as providing pseudo-counts
for a latent “visual groundedness” probability. Crucially,
this acts as a trust mechanism rather than an error detector:
• High alignment (st ↑): Indicates the generation is
consistent with visual evidence (high trust). The gate
increases to reinforce this valid trajectory.
• Low/Negative alignment (st ↓): Indicates uncertainty
or non-visual context. The gate suppresses intervention
to prevent over-steering and preserve fluency.
This monotone design is intentional. A high value of st
means the current hidden state is already compatible with the
extracted visual evidence, so reinforcing vCARD is unlikely
to disrupt generation. Conversely, a low or negative value
of st suggests either non-visual function-token generation
or an unstable trajectory, where aggressive steering can
hurt fluency or object recall. The gate therefore controls
trust in the current visual alignment, rather than measuring
deviation magnitude.
Gate Formulation.
Using this intuition, we calculate the
gate parameters using a smooth mapping from st:
αt = softplus(k st + c),
βt = softplus(−k st + c),
gt =
αt
αt + βt
.
(3)
Here, k is a sensitivity hyperparameter controlling the steep-
ness of the response, while c is a concentration parameter
typically fixed to a robust default (e.g., c = 1). We opti-
mize these few parameters on a small held-out validation
set (details in Sec. 4.1). To ensure stability, we clamp the
gate’s output to a predefined range, gt ∈[gmin, gmax]. This
prevents the gate from completely shutting off (gt→0) or sat-
urating (gt→1) too readily, making the intervention robust
to noise.
For generating each token t in the answer, the final steering
update vsteer
t
combines the adaptive gate with a global cap
αmax:
vsteer
t
=
 αmax gt

|
{z
}
adaptive strength
vCARD,
(4)
This vector is injected into the residual stream immediately
after the Self-Attention(SA) operation. The updated hidden
state, hnew
l,t , is thus computed as:
hnew
l,t
=
 hl,t + SA(hl,t)

+ vsteer
t
.
(5)
The term αmaxgt represents the adaptive strength of the
intervention, ensuring a strong corrective signal is applied
only when needed; the injection is restricted to the answer
span.
3.4. Integration and Workflow
Our complete method, Residual-Update Directed
DEcoding Regulation (RUDDER), integrates the
CARD vector and the adaptive Beta Gate to mitigate hal-
lucination by steering LVLMs toward visually grounded
outputs. We refer to this primary, adaptive configuration
as RUDDER-Beta. To isolate the impact of adaptivity, we
also define a static variant, RUDDER-Add, which injects
the CARD vector with a constant strength without the gating
mechanism.
As detailed in Algorithm 1, RUDDER can be seamlessly in-
tegrated into the standard auto-regressive decoding loop. By
operating within a single inference pass, it mitigates hallu-
cination with negligible computational overhead, resolving
the common trade-off between efficacy and efficiency. The
overall workflow of this approach is illustrated in Figure 2.
4. Experiments
In this section, we validate RUDDER, demonstrating its
ability to mitigate hallucination effectively with negligible
computational overhead. We conduct a series of experi-
ments across diverse LVLM architectures and benchmarks
to assess performance, general capabilities, efficiency, and
hyperparameter sensitivity.
4.1. Experimental Setup
Model Architectures. We evaluate RUDDER on three
representative LVLMs with distinct visual-textual align-
ment mechanisms: LLaVA-1.5–7B (Liu et al., 2024a) and
Idefics2–8B (Laurenc¸on et al., 2024) (linear projection),
and InstructBLIP (Dai et al., 2023) (Q-former). We also in-
clude LLaVA-1.5-13B (Liu et al., 2024a) and Qwen2.5-VL-
7B (Bai et al., 2025) in our scalability analysis (Sec. 4.6).
Baselines.
We compare RUDDER against state-of-the-
art inference-time intervention methods. These include
logit-based strategies like DoLa (Chuang et al., 2023),
4

## Page 5

Adaptive Residual-Update Steering for Hallucination Mitigation
Pool
Normalize
∆𝑖= 𝐴𝑖
𝑣𝑡
𝑠𝑡𝑒𝑒𝑟
Cosine 
similarity
Softplus
Self-Attn
Layer Norm
FFNN
Layer Norm
+
Decoder Layers
Decoder Layers
Layer 𝑙
∆𝑖
𝑙= 𝐴𝑖
𝑙
Self-Attn
Layer Norm
FFNN
Layer Norm
+
Decoder Layers
Decoder Layers
Layer 𝑙
Prefill Stage
Decoding Stage
Beta Gate
clamp
ℎ𝑙,𝑡
Cache
𝑣CARD
ℎ𝑙,𝑡
𝑣CARD
Figure 2. The overall workflow of RUDDER. (1) Prefill Stage (Yellow): We extract the CARD vector vCARD by aggregating residual
updates from the target layer. (2) Decoding Stage (Orange): At each step t, the Adaptive Beta Gate computes a steering vector vsteer
t
to
modulate the residual stream, guiding the model toward visually grounded outputs.
VCD (Leng et al., 2023), PAI (Liu et al., 2024b), and the
recent HALC (Chen et al., 2024), AGLA (An et al., 2025),
and DeGF (Zhang et al., 2025); as well as steering-based in-
terventions like VISTA (Li et al., 2025). All baseline results
were reproduced under identical evaluation settings.
Implementation Details. We configure RUDDER using a
lightweight calibration on 100 held-out MSCOCO images,
disjoint from the evaluation subset. This calibration is used
only to select deployment hyperparameters and is not a
training procedure. The process involves two main steps:
• Step 1: Layer Selection (L). We perform a linear sweep
to maximize CARD vector alignment. We select mid-
to-late layers for LLaVA-1.5 (L = 30) and Idefics2
(L = 28), and an early layer for InstructBLIP (L = 1).
• Step 2: Hyperparameter Tuning. We grid-search the
steering strength αmax and gate sensitivity k to mini-
mize CHAIR scores while maintaining ≥95% base-
line recall. Other parameters are fixed to robust defaults
(c = 1, g ∈[0.05, 1]).
Final settings (αmax, k) are (20, 5.0) for LLaVA-1.5,
(8.0, 5.0) for Idefics2, and (6.5, 8.0) for InstructBLIP. This
one-time setup incurs negligible overhead. We deliberately
constrain the search to preserve at least 95% of the vanilla
recall, since hallucination reduction is only useful when the
model does not lose its general ability to mention correct
objects.
Evaluation Benchmarks.
We evaluate RUDDER us-
ing both specialized hallucination benchmarks and a
general capability benchmark.
(1) CHAIR (Rohrbach
et al., 2019):
We evaluate open-ended captioning on
MSCOCO (Lin et al., 2015) validation set, reporting
sentence-level (CHAIRS) and object-level (CHAIRI) met-
rics. For the CHAIR benchmark, we evaluate 500 samples
using the prompt “Please help me describe the
image in detail” with a maximum length of 512 to-
kens. (2) POPE (Li et al., 2023): We assess object probing
via targeted yes/no questions, reporting Accuracy and F1
scores averaged across random, popular, and adversarial
splits. For the POPE benchmark, we use the prompt “Is
there a <object> in the image?” and let the
models to answer with “yes” or “no”. (3) MME (Fu et al.,
2024): We use MME to confirm that our method maintains
general multimodal capabilities (e.g., reasoning, counting).
Following established protocols, we set the temperature to
1.0 and test greedy, beam search (beam=5), and nucleus
sampling (p = 0.9).
4.2. Results on Hallucination Benchmarks
4.2.1. CHAIR: OPEN-ENDED CAPTIONING
On the CHAIR benchmark, which evaluates hallucination
in open-ended captioning, RUDDER demonstrates a strong
ability to reduce factual errors while preserving caption
quality.
A key challenge in hallucination mitigation is the trade-
off with recall: aggressive steering can artificially lower
hallucination scores by producing overly simplistic captions.
To ensure a fair and practical evaluation, we constrain our
analysis to configurations that maintain at least 95% of the
vanilla model’s recall (i.e., Recall{evaluated methods} ≥0.95 ×
Recall{vanilla model}).
Under this constraint, RUDDER-Beta consistently outper-
5

## Page 6

Adaptive Residual-Update Steering for Hallucination Mitigation
forms the vanilla baseline across all tested LVLMs and de-
coding strategies, as shown in Table 1. It achieves average
relative reductions of 33.2% in sentence-level (CHAIRS)
and 28.6% in object-level (CHAIRI) hallucination across
these three models.
Compared to strong baselines like VCD and DoLa, our
method is consistently superior on both metrics. Further-
more, RUDDER-Beta performs on par with the state-of-the-
art VISTA and, on average, yields a greater reduction in
object-level hallucinations (CHAIRI).
RUDDER-Beta’s ability to reduce CHAIRI more effectively
than CHAIRS highlights its precision. We attribute this to
the token-wise gating mechanism, which selectively am-
plifies corrections on visually incongruent or content-noun
tokens while leaving already grounded tokens largely un-
perturbed. This allows RUDDER to preferentially suppress
object-level hallucinations without degrading overall cap-
tion quality and recall.
4.2.2. POPE: VISUAL QUESTION ANSWERING
Moving from open-ended captioning to a more constrained
task, we next evaluate RUDDER on the POPE benchmark
for object probing. This benchmark tests the model’s fac-
tuality through targeted yes/no questions, offering a differ-
ent perspective on hallucination. In this setting, RUDDER
again demonstrates competitive performance. As shown
in Table 2, RUDDER consistently outperforms the vanilla
baselines and most competing methods across all tested
models. Concretely, RUDDER-Beta improves accuracy by
1.0/0.7/0.5 absolute points (pp) and F1 by 1.6/1.3/0.14
pp on LLaVA-1.5, Idefics2, and InstructBLIP, respectively.
Notably, RUDDER-Beta achieves the highest F1-score and
accuracy on both LLaVA-1.5 and Idefics2, surpassing strong
steering-based methods like VISTA. While its performance
on InstructBLIP is slightly surpassed by VISTA when em-
ploying greedy decoding and nucleus sampling, RUDDER
remains highly competitive, highlighting its effectiveness as
a versatile solution for reducing object hallucination.
4.2.3. ANALYSIS OF ADAPTIVE VS. FIXED-STRENGTH
STEERING
A key design choice in RUDDER is the regulation of injec-
tion strength. We analyze the trade-off between the adap-
tive gate (RUDDER-Beta) and the fixed-strength injection
(RUDDER-Add). Our experiments show a clear distinction
between these variants, guiding the choice based on the task
and model architecture.
For complex, open-ended generation (CHAIR), RUDDER-
Beta is consistently superior. Its token-wise precision is
crucial for suppressing specific hallucinations in long-form
text without harming overall recall. In the simpler, binary-
choice POPE task, the distinction is more nuanced. While
RUDDER-Beta remains the top performer on LLaVA-1.5
and Idefics2, RUDDER-Add is competitive and even sur-
passes RUDDER-Beta on InstructBLIP. We hypothesize
this is partly because InstructBLIP’s Q-Former provides a
highly-condensed visual representation that responds well
to a uniform steering signal in a simple setting. For single-
token “yes/no” answers, the aggressive push from fixed-
strength steering can be sufficient and sometimes more ben-
eficial for certain model architectures.
In summary, RUDDER-Beta is recommended for robust
and precise control in complex tasks, while the simpler
RUDDER-Add is a powerful option for constrained tasks
and certain model architectures.
4.3. Results on Comprehensive Benchmarks
To ensure that hallucination mitigation does not compro-
mise general multimodal capabilities, we evaluate RUD-
DER on MME benchmark. The results show that RUDDER
successfully reduces hallucinations without sacrificing the
overall abilities of the tested LVLMs. As demonstrated in
Table 3, both RUDDER-Beta and RUDDER-Add achieve
higher MME scores than the vanilla models for Idefics2
and InstructBLIP. On LLaVA-1.5, RUDDER’s scores are
slightly lower than the vanilla model, but the difference is
still acceptable.
4.4. Efficiency Tests
A critical advantage of RUDDER is its low computational
overhead, making it practical for real-world deployment.
Unlike many state-of-the-art intervention methods that re-
quire extra forward passes and significantly increase latency,
RUDDER is designed to operate within a single generative
pass. We measure the practical latency and throughput of
RUDDER against vanilla models and other methods, with
results presented in Table 4. All experiments are conducted
on a single Nvidia A100 GPU with 80 GB VRAM and
a batch size fixed at 1. RUDDER-Beta maintains an av-
erage throughput of 96.0% compared to vanilla LVLMs.
RUDDER-Add is even more efficient as it bypasses the
Beta Gate calculation. In contrast, competing methods that
require extra forward passes see a significant drop in effi-
ciency. On average, the throughput of a method like VISTA
is only 58.1% of the vanilla models.
4.5. Ablation Studies
We conduct an ablation study on Idefics2 using the CHAIR
benchmark to analyze the key hyperparameters: injection
layer L, maximum steering strength αmax and the gate sen-
sitivity k.
We first identify the optimal intervention layer, finding Layer
6

## Page 7

Adaptive Residual-Update Steering for Hallucination Mitigation
Table 1. Hallucination evaluation on the CHAIR benchmark. We compare RUDDER against state-of-the-art training-free methods
with a max generation length of 512 tokens. Note that methods marked with † (e.g., DeGF) require iterative updates, incurring significantly
higher latency (> 3×) compared to RUDDER’s single-pass efficiency. Best results are bolded and second-best are underlined.
Decoding
Method
LLaVA-1.5 (Liu et al., 2024a)
Idefics2 (Laurenc¸on et al., 2024)
InstructBLIP (Dai et al., 2023)
CHAIRS ↓
CHAIRI ↓
CHAIRS ↓
CHAIRI ↓
CHAIRS ↓
CHAIRI ↓
Greedy
Vanilla
48.6
13.6
46.6
14.9
39.2
12.8
DoLa (Chuang et al., 2023)
47.6
13.4
-
-
-
-
VCD (Leng et al., 2023)
49.8
14.5
-
-
46.4
15.3
HALC (Chen et al., 2024)
40.3
10.3
-
-
-
-
AGLA (An et al., 2025)
44.6
12.1
-
-
-
-
DeGF† (Zhang et al., 2025)
34.2
9.2
-
-
-
-
VISTA (Li et al., 2025)
38.6
11.4
33.5
11.6
27.7
9.7
RUDDER-Beta (Ours)
39.5
10.5
28.4
10.9
27.1
8.5
RUDDER-Add (Ours)
42.1
11.8
30.1
11.8
28.3
10.4
Beam Search
Vanilla
52.8
15.6
48.6
14.5
38.2
12.7
VCD (Leng et al., 2023)
52.4
15.5
-
-
47.4
16.3
VISTA (Li et al., 2025)
33.9
10.5
32.2
11.8
27.1
9.6
RUDDER-Beta (Ours)
33.1
9.3
29.2
10.1
26.2
9.5
RUDDER-Add (Ours)
35.2
10.6
31.4
10.9
27.4
11.1
Nucleus
Sampling
Vanilla
55.6
16.0
53.8
16.7
46.0
16.2
DoLa (Chuang et al., 2023)
49.3
14.8
-
-
-
-
VCD (Leng et al., 2023)
57.5
17.2
-
-
53.3
19.8
VISTA (Li et al., 2025)
39.2
11.9
35.5
11.8
29.0
11.3
RUDDER-Beta (Ours)
39.9
11.0
34.1
11.3
28.9
13.7
RUDDER-Add (Ours)
41.6
12.1
36.5
12.9
30.1
14.4
Table 2. Performance on the POPE benchmark across three LVLMs. The reported values are the mean accuracy and F1 score,
aggregated over the random, popular, and adversarial object splits. The best scores are bolded, and the second best scores are underlined.
Decoding
Method
LLaVA-1.5 (Liu et al., 2024a)
Idefics2 (Laurenc¸on et al., 2024)
InstructBLIP (Dai et al., 2023)
Acc ↑
F1 ↑
Acc ↑
F1 ↑
Acc ↑
F1 ↑
Greedy
Vanilla
85.34
84.91
78.40
74.86
85.74
84.75
DoLa (Chuang et al., 2023)
85.51
84.96
-
-
-
-
VCD (Leng et al., 2023)
85.46
84.87
-
-
85.79
84.89
PAI (Liu et al., 2024b)
85.98
85.31
-
-
-
-
VISTA (Li et al., 2025)
86.21
85.42
78.28
74.66
86.25
85.06
RUDDER-Beta (Ours)
86.53
86.03
78.74
76.52
86.02
84.93
RUDDER-Add (Ours)
85.92
84.98
78.43
75.91
86.05
85.05
Beam Search
Vanilla
85.46
84.98
78.67
77.55
84.73
84.37
VCD (Leng et al., 2023)
85.60
85.06
-
-
84.95
84.59
PAI (Liu et al., 2024b)
85.58
85.01
-
-
-
-
VISTA (Li et al., 2025)
86.10
85.35
78.40
77.31
85.64
84.61
RUDDER-Beta (Ours)
86.51
86.19
79.33
77.96
85.54
84.40
RUDDER-Add (Ours)
85.98
85.02
78.91
77.60
85.71
84.75
Nucleus
Sampling
Vanilla
83.00
81.08
74.84
67.78
85.50
84.52
DoLa (Chuang et al., 2023)
82.94
81.12
-
-
-
-
VCD (Leng et al., 2023)
82.82
81.90
-
-
85.61
84.65
PAI (Liu et al., 2024b)
83.17
82.14
-
-
-
-
VISTA (Li et al., 2025)
83.58
82.21
74.66
67.70
86.12
85.26
RUDDER-Beta (Ours)
84.02
83.57
75.89
69.69
85.79
84.74
RUDDER-Add (Ours)
83.20
82.38
74.95
67.84
85.95
84.95
28 is the most effective for the Idefics2 model, as shown in
Figure 3. Focusing on this layer, we then tune the hyperpa-
rameters αmax and k. The heatmaps in Figures 4a through
4c reveal a core trade-off: increasing the steering strength
(αmax) effectively reduces CHAIR scores but at the cost of
lower recall. The gate sensitivity k, does not exhibit a sim-
ple linear trend; instead, it plays a crucial modulating role in
this trade-off. Ultimately, we find that the best balance for
Idefics2 is achieved with αmax = 8.0 and k = 5.0. Ablation
results for other models are presented in Appendix B.2.
7

## Page 8

Adaptive Residual-Update Steering for Hallucination Mitigation
Table 3. Overall performance scores on the MME full evalu-
ation set. Higher scores indicate better general capability across
perception, reasoning, and knowledge-based tasks.
Decoding
Method
LLaVA-1.5
Idefics2
InstructBLIP
Greedy
Vanilla
1745.87
1518.84
1566.77
RUDDER-Beta
1724.17
1540.56
1592.07
RUDDER-Add
1715.45
1526.03
1585.28
Beam Search
Vanilla
1760.20
1450.59
1539.16
RUDDER-Beta
1746.66
1484.21
1565.77
RUDDER-Add
1738.13
1475.80
1560.64
Nucleus
Sampling
Vanilla
1752.65
1362.45
1538.18
RUDDER-Beta
1721.94
1374.77
1556.43
RUDDER-Add
1713.74
1364.16
1546.01
Table 4. Throughput Comparison (tokens/s). Measurements are
conducted using greedy decoding. Higher scores indicate better
efficiency.
Method
LLaVA-1.5
Idefics2
InstructBLIP
Vanilla
56.7
47.8
62.3
VCD
30.1
-
-
PAI
29.5
-
-
VISTA
36.1
31.9
28.9
RUDDER-Beta (Ours)
54.9
45.8
59.5
RUDDER-Add (Ours)
55.8
46.5
60.8
4.6. Scalability and Generalization Analysis
To verify that RUDDER is not over-fitted to specific model
sizes or architectures, we extend our evaluation to LLaVA-
1.5-13B (to verify scalability) and the recent Qwen2.5-VL-
7B (to verify architectural generalization).
As shown in Table 5, RUDDER demonstrates robust perfor-
mance across these diverse settings. On the larger LLaVA-
1.5-13B, RUDDER scales effectively, surpassing the VISTA
baseline in sentence-level hallucination reduction (CHAIRS:
39.9 vs. 40.1) and achieving the highest POPE performance
(F1: 85.5).
The results on Qwen2.5-VL highlight the precision of our
method. While VISTA achieves a lower CHAIRS, it suffers
a regression in POPE performance compared to the vanilla
baseline (F1: 87.8 →87.4, Acc: 88.8 →88.6), suggesting
that its aggressive steering may compromise the model’s
fundamental object recognition capabilities. In contrast,
RUDDER achieves the lowest instance-level hallucination
rate (CHAIRI: 7.0) while simultaneously improving POPE
performance (F1: 88.1). This confirms that the CARD vec-
tor captures a universal and precise visual evidence signal,
mitigating hallucinations without the side effect of degrad-
ing general recognition accuracy.
4.7. Case Study
Qualitative analysis in Appendix B.3 demonstrates RUD-
DER’s effectiveness. The case studies show that RUD-
26
27
28
29
30
Layer
0
10
20
30
40
Score (%) 
 lower is better
37.2
44.6
28.4
29.4
35.0
12.1
11.8
10.9
10.2
11.8
Layer CHAIRs & CHAIRi @ 
max=8.0, K=5.0
CHAIRs (
)
CHAIRi (
)
Figure 3. Layer ablation study on Idefics2. We fix (αmax =
8.0, k = 5.0) and vary the injection layer. Mid–late layers
(L ≈28–30) prove most effective, with L=28 yielding a strong
reduction.
Table 5. Scalability and Generalization Analysis. We assess
RUDDER on a larger backbone (LLaVA-1.5-13B) and a distinct
modern architecture (Qwen2.5-VL-7B) using Greedy decoding.
Model
Method
CHAIR
POPE
CS↓
CI↓
Acc↑
F1↑
LLaVA-1.5
(13B)
Vanilla
44.2
11.8
85.7
85.0
VISTA
40.1
10.6
86.0
85.2
RUDDER
39.9
10.8
86.3
85.5
Qwen2.5-VL
(7B)
Vanilla
35.2
9.5
88.8
87.8
VISTA
25.1
7.7
88.6
87.4
RUDDER
26.9
7.0
89.0
88.1
DER not only eliminates object hallucinations present in the
vanilla model’s outputs but also produces more conservative
content. By avoiding the vanilla model’s confident yet in-
correct assertions, RUDDER enhances the model’s overall
reliability.
5. Conclusion and Limitations
In this work, we introduce RUDDER, a low-overhead
inference-time intervention framework that mitigates
LVLMs hallucination using two key innovations: the zero-
cost CARD vector, which extracts a per-sample visual
evidence from the model’s own residual updates, and the
adaptive Beta Gate, which applies a corrective signal
with principled, token-wise strength.
Experiments confirm RUDDER achieves state-of-the-art
comparable performance on benchmarks like CHAIR and
POPE with negligible computational overhead, resolving the
common efficacy-efficiency trade-off. RUDDER presents a
practical and effective solution for enhancing the reliability
of LVLMs in real-world settings.
RUDDER’s primary limitation is its sensitivity to hyperpa-
rameters, which need be tuned for each model architecture.
Future work could focus on automated hyperparameter opti-
mization to improve its robustness and ease of deployment.
8

## Page 9

Adaptive Residual-Update Steering for Hallucination Mitigation
0.0
4.0
8.0
12.0
16.0
20.0
Strength 
max
0.0
3.0
5.0
8.0
10.0
Sensitivity K
46.4
33.4
32.4
20.6
3.4
0.6
46.4
36.4
32.2
25.6
9.4
1.0
46.4
33.8
28.4
27.4
13.4
2.4
46.4
33.6
32.8
32.0
14.2
8.0
46.4
33.8
33.2
27.6
14.4
10.4
CHAIRs (
) 
 layer 28  (value: %)
10
20
30
40
Value (%)
(a) CHAIRS heatmap
0.0
4.0
8.0
12.0
16.0
20.0
Strength 
max
0.0
3.0
5.0
8.0
10.0
Sensitivity K
14.7
11.0
13.1
10.6
24.6
4.6
14.7
11.7
10.2
12.9
15.7
3.4
14.7
11.2
10.9
9.5
7.5
5.7
14.7
11.9
11.4
12.9
6.0
7.6
14.7
11.9
10.4
10.7
6.6
5.8
CHAIRi (
) 
 layer 28  (value: %)
5.0
7.5
10.0
12.5
15.0
17.5
20.0
22.5
Value (%)
(b) CHAIRi heatmap
0.0
4.0
8.0
12.0
16.0
20.0
Strength 
max
0.0
3.0
5.0
8.0
10.0
Sensitivity K
60.3
56.8
57.4
50.3
15.5
3.8
60.3
58.1
57.7
55.2
35.4
9.9
60.3
58.4
57.8
55.1
43.6
17.4
60.3
57.8
57.6
56.8
50.2
35.2
60.3
58.4
58.0
56.4
51.9
43.2
Recall ( ) 
 layer 28  (value: %)
10
20
30
40
50
60
Value (%)
(c) Recall heatmap
Figure 4. Hyperparameter sensitivity analysis on Idefics2. We analyze the trade-off between steering strength αmax and gate
sensitivity k at Layer 28. Stronger steering reduces hallucinations (a, b) but may impact recall (c). The optimal balance is found around
αmax = 8.0, k = 5.0.
Impact Statement
This paper presents RUDDER, a method designed to miti-
gate object hallucinations in LVLMs. Hallucinations pose
significant risks in real-world applications, such as au-
tonomous systems, medical imaging analysis, and robotics,
where factual errors can lead to safety hazards. By improv-
ing the grounding of model outputs in visual evidence, our
work contributes to the development of more reliable and
trustworthy AI systems.
A core contribution of RUDDER is its computational effi-
ciency. RUDDER operates with negligible overhead. This
efficiency aligns with the goals of “Green AI,” reducing the
energy footprint of high-quality model inference and low-
ering the barrier to deploying aligned LVLMs in resource-
constrained environments.
Acknowledgements
This work was supported by the Research Council of Fin-
land (Flagship programme: Finnish Center for Artificial
Intelligence FCAI, and grant 358246).
References
An, W., Tian, F., Leng, S., Nie, J., Lin, H., Wang, Q., Chen,
P., Zhang, X., and Lu, S. Mitigating object hallucinations
in large vision-language models with assembly of global
and local attention. In 2025 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp.
29915–29926. IEEE, 2025.
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang,
K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y.,
Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu,
Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang,
Z., Xu, H., and Lin, J. Qwen2.5-vl technical report, 2025.
URL https://arxiv.org/abs/2502.13923.
Chen, Z., Zhao, Z., Luo, H., Yao, H., Li, B., and Zhou,
J. HALC: Object hallucination reduction via adaptive
focal-contrast decoding. In Forty-first International Con-
ference on Machine Learning, 2024.
URL https:
//openreview.net/forum?id=EYvEVbfoDp.
Chuang, Y.-S., Xie, Y., Luo, H., Kim, Y., Glass, J., and
He, P. Dola: Decoding by contrasting layers improves
factuality in large language models.
arXiv preprint
arXiv:2309.03883, 2023.
Dai, W., Li, J., Li, D., Tiong, A. M. H., Zhao, J., Wang,
W., Li, B., Fung, P., and Hoi, S. InstructBLIP: Towards
General-purpose Vision-Language Models with Instruc-
tion Tuning, June 2023. URL http://arxiv.org/
abs/2305.06500. arXiv:2305.06500 [cs].
Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang,
J., Zheng, X., Li, K., Sun, X., Wu, Y., and Ji, R. MME: A
Comprehensive Evaluation Benchmark for Multimodal
Large Language Models, March 2024. URL http://
arxiv.org/abs/2306.13394. arXiv:2306.13394
[cs].
Hokamp, C. and Liu, Q. Lexically constrained decoding for
sequence generation using grid beam search, 2017. URL
https://arxiv.org/abs/1704.07138.
Laurenc¸on, H., Tronchon, L., Cord, M., and Sanh, V. What
matters when building vision-language models?
Ad-
vances in Neural Information Processing Systems, 37:
87874–87907, 2024.
Leng, S., Zhang, H., Chen, G., Li, X., Lu, S., Miao, C.,
and Bing, L. Mitigating Object Hallucinations in Large
Vision-Language Models through Visual Contrastive De-
coding, November 2023. URL http://arxiv.org/
abs/2311.16922. arXiv:2311.16922 [cs].
Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., and Wen, J.-R.
Evaluating object hallucination in large vision-language
9

## Page 10

Adaptive Residual-Update Steering for Hallucination Mitigation
models. In Proceedings of the 2023 conference on empiri-
cal methods in natural language processing, pp. 292–305,
2023.
Li, Z., Shi, H., Gao, Y., Liu, D., Wang, Z., Chen, Y., Liu, T.,
Zhao, L., Wang, H., and Metaxas, D. N. The hidden life of
tokens: Reducing hallucination of large vision-language
models via visual information steering. Proceedings of
Machine Learning Research, 267:35799–35819, 2025.
Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick,
R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L.,
and Doll´ar, P. Microsoft coco: Common objects in con-
text, 2015. URL https://arxiv.org/abs/1405.
0312.
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines
with visual instruction tuning. In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pp. 26296–26306, 2024a.
Liu, S., Zheng, K., and Chen, W. Paying more attention
to image: A training-free method for alleviating halluci-
nation in lvlms. In European Conference on Computer
Vision, pp. 125–140. Springer, 2024b.
Liu, S., Ye, H., and Zou, J. Reducing hallucinations in
large vision-language models via latent space steering.
In The Thirteenth International Conference on Learning
Representations, 2025. URL https://openreview.
net/forum?id=LBl7Hez0fF.
Manakul, P., Liusie, A., and Gales, M. J. F. Selfcheck-
gpt: Zero-resource black-box hallucination detection for
generative large language models, 2023. URL https:
//arxiv.org/abs/2303.08896.
Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T.,
and Saenko, K. Object hallucination in image caption-
ing, 2019. URL https://arxiv.org/abs/1809.
02156.
Sensoy, M., Kaplan, L., and Kandemir, M. Evidential Deep
Learning to Quantify Classification Uncertainty, Octo-
ber 2018.
URL http://arxiv.org/abs/1806.
01768. arXiv:1806.01768 [cs].
Song, K., Jang, J., Shin, S. j., and Moon, I.-C. Bivariate
Beta-LSTM, November 2019. URL http://arxiv.
org/abs/1905.10521. arXiv:1905.10521 [cs].
Su, J., Chen, J., Li, H., Chen, Y., Qing, L., and Zhang,
Z. Activation Steering Decoding: Mitigating Hallucina-
tion in Large Vision-Language Models through Bidirec-
tional Hidden State Intervention. In Che, W., Nabende,
J., Shutova, E., and Pilehvar, M. T. (eds.), Proceed-
ings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp.
12964–12974, Vienna, Austria, July 2025. Association
for Computational Linguistics. ISBN 979-8-89176-251-
0. doi: 10.18653/v1/2025.acl-long.634. URL https:
//aclanthology.org/2025.acl-long.634/.
Yamada, Y., Lindenbaum, O., Negahban, S., and Kluger,
Y. Feature selection using stochastic gates. In Interna-
tional conference on machine learning, pp. 10648–10659.
PMLR, 2020.
Zhang, C., Wan, Z., Kan, Z., Ma, M. Q., Stepputtis, S.,
Ramanan, D., Salakhutdinov, R., Morency, L.-P., Sycara,
K. P., and Xie, Y. Self-correcting decoding with genera-
tive feedback for mitigating hallucinations in large vision-
language models. In The Thirteenth International Confer-
ence on Learning Representations, 2025. URL https:
//openreview.net/forum?id=tTBXePRKSx.
Zhao, L., Deng, Y., Zhang, W., and Gu, Q.
Mitigat-
ing object hallucination in large vision-language mod-
els via image-grounded guidance, 2025. URL https:
//openreview.net/forum?id=eFoj2egr7G.
A. Additional Illustration on the Methodology
A.1. RUDDER Algorithm
Here we present the pseudo-code for RUDDER illustrated
in Sec. 3.
A.2. From Na¨ıve Bayes to the Adaptive Gate
Problem setup.
At each decoding step t, we want a
scalar gate gt ∈(0, 1) that reflects how much the cur-
rent token should be nudged toward the visual evidence
direction vCARD.
Let the alignment statistic be st =
cos(ht, vCARD) ∈[−1, 1].
Na¨ıve Bayes view (posterior as a gate).
Introduce a latent
Bernoulli variable Zt ∈{0, 1} indicating whether the token
is visually grounded (Zt = 1) or at risk of drifting (Zt = 0).
We use the posterior mean gt = E[Zt | st] as a continuous
gate (rather than a hard on/off decision).
Beta–Bernoulli conjugacy with “soft counts”.
With a
Beta prior Beta(αt, βt) on Zt, the posterior mean is
gt =
αt
αt + βt
.
We map the alignment st to positive pseudo-counts via a
smooth, monotone transform:
αt = softplus(k st + c),
βt = softplus(−k st + c),
where k controls sensitivity and c controls concentra-
tion/bias. The softplus ensures strictly positive, numerically
stable “counts”.
10

## Page 11

Adaptive Residual-Update Steering for Hallucination Mitigation
Algorithm 1 RUDDER (single-pass, test-time steering; fixed target layer ℓ)
1: Input: Model M; image ximg, text xtext; fixed layer ℓ; hyperparams (αmax, k, c, gmin, gmax)
2: Tpre ←TokenizePrefill(ximg, xtext) {image + prompt tokens}
3: Prefill: run M once (read-only hook at layer ℓ) to build KV cache and cache {Aℓ
i}i∈Tpre
4: ∆ℓ
i ←Aℓ
i by Eq. 1 {pre-norm: residual update equals SA output}
5: vCARD ←by Eq. 2 {Pool →L2-Normalize over {∆ℓ
i}i∈Tpre}
6: Decode: for t = 1, 2, . . . {auto-regressive generation}
7:
st ←cos
 hℓ,t, vℓ
CARD

8:
(αt, βt, gℓ,t) ←by Eq. 3
9:
gℓ,t ←clip(gℓ,t, gmin, gmax)
10:
hnew
ℓ,t
←
 hℓ,t + SA(ℓ)(hℓ,t)

+ 1[t ∈Tans] · vsteer
t
{post-SA residual add; answer span only}
11:
emit next token
12: end for
Properties (useful for calibration).
The resulting gt is
monotone in st, symmetric g(−s) = 1−g(s), and bounded
in (0, 1). Around s = 0, the slope
∂g
∂s

s=0
=
k σ(c)
2 softplus(c),
σ(x) =
1
1 + e−x ,
gives a handy knob to set how fast the gate reacts to align-
ment changes.
Stability: clamping and optional per-token cap.
For
robustness we clamp gt ←clip(gt; gmin, gmax) to avoid both
shutting off (gt →0) and saturating (gt →1). Optionally, we
enforce a per-token norm cap τ:
αmaxgt bvCARD

2 ≤τ,
bv = v/∥v∥2,
which further prevents rare spikes when hidden-state norms
vary.
Final update (matches Algorithm 1).
vsteer
t
=
 αmax gt

|
{z
}
adaptive strength
vCARD
(6)
hnew
l,t
=
 hl,t + SA(hl,t)

+ vsteer
t
(7)
We apply this only on the answer span using the mask mt
as in Algorithm 1.
Implementation notes.
• We compute st with L2-normalized ht and vCARD (co-
sine similarity).
• gt is clamped to [gmin, gmax]; the global scale αmax
controls the maximal push.
• vCARD is extracted once during the mandatory prefill
pass (zero extra forwards).
Practical calibration recipe.
Choose c to set the overall
smoothness (typical c∈[0.5, 1.5]), then increase k until gt
becomes sufficiently responsive on a small dev set. Finally
tune αmax and [gmin, gmax] for stability/strength trade-offs.
Complexity.
The gate requires only light-weight vector
ops during decoding and reuses the prefill to compute vCARD.
Hence no extra forward pass compared to vanilla generation.
A.3. Clarification
RUDDER maintains the same inference efficiency as the
original model, requiring no extra forward passes. The
CARD vector is extracted opportunistically during the
mandatory prefill pass, and the steering is applied within
each step of the subsequent decoding pass.
By no extra forward pass we mean no additional
model.forward invocations beyond the vanilla prefill and
decode; our overhead comes only from cheap per-token
vector operations implemented via hooks.
A.4. Visualization and Quantification of the Geometry
of vCARD and vsteer
t
Setup and link to motivation.
Large VLMs fuse vision
and text via self-attention; the residual update of this sub-
layer thus captures the net impact of visual context on
token representations. Motivated by this, we aggregate
the prefill-phase residual updates to obtain a per-sample
direction vCARD ∈Rd and define its steering counterpart
vsteer
t
=(αmax gate) vCARD (Sec. 3). For each image we ex-
port (i) vCARD (image+prompt) and its text-only variant, and
(ii) vsteer
t
. We reduce vectors by PCA (k=50) then t-SNE
(default perplexity unless noted), and cluster the steering
space with KMeans (best silhouette over K ∈{2, . . . , 10}).
Figure 5 shows two key views: a paired overlay of im-
age+text vs. text-only vCARD with one-to-one lines, and the
clustered t-SNE of vsteer
t
. The overlay reveals systematic
sample-wise rotations from the language prior (text-only) to
11

## Page 12

Adaptive Residual-Update Steering for Hallucination Mitigation
40
20
0
20
40
60
t-SNE 1
15
10
5
0
5
10
15
20
t-SNE 2
vcard (img+text) overlaid with vcard (text-only)
C0 (n=78)
C1 (n=22)
card_text_only
(a) t-SNE overlay of vCARD
(img+text;
colored by vsteer
t
clusters) and text-only vCARD
(black), with lines linking paired
samples.
10
5
0
5
10
t-SNE 1
15
10
5
0
5
t-SNE 2
vsteer (t-SNE)
C0 (n=78)
C1 (n=22)
(b) t-SNE of vsteer
t
with KMeans
clusters.
Figure 5. Structure in steering space (b) and its sample-wise
projection to vCARD (a).
the image-conditioned direction, and these rotations point
toward coherent steering clusters—visual evidence is there-
fore directional rather than noise, directly supporting our
motivation.
Quantifying directional structure.
We quantify two ef-
fects central to our hypothesis: (i) Rotation from text-only
to image-conditioned CARD: ∆θ = arccos⟨vtext, vimg+txt⟩
shows a tight distribution around ∼40◦(mean ≈40.5◦, me-
dian ≈40.4◦), indicating a consistent, non-trivial visual-
induced rotation rather than random drift (Fig. 6a); this
effect persists across steering clusters (Fig. 6c). (ii) Align-
ment gain w.r.t. steering: ⟨vimg+txt, vsteer⟩−⟨vtext, vsteer⟩
is positive on average (mean ≈0.239, median ≈0.238) and
remains positive across clusters (Figs. 6b,d), showing that
image-conditioned vCARD is closer to the actual steering
geometry used by the β-gate. Together, these results sub-
stantiate our motivation: aggregating self-attention residual
updates yields a robust sample-specific visual-evidence di-
rection that aligns with the downstream steering mechanism.
Notes.
Silhouette scores are typically higher for vsteer
t
than
for vCARD, consistent with the gate organizing/scaling direc-
tions across samples. We emphasize that t-SNE primarily
supports local neighborhood interpretation; all scalar statis-
tics are computed in the original vector spaces.
Semantic clustering of CARD.
In addition to the paired
geometry above, we directly test whether CARD encodes
category-level visual semantics. We extract CARD vectors
from LLaVA-1.5-7B on four COCO categories with 50 im-
ages per category and visualize them with t-SNE. Figure 7
shows that samples from the same visual category form
coherent neighborhoods, suggesting that CARD captures
image-conditioned semantic structure rather than random
activation noise.
39.0
39.5
40.0
40.5
41.0
41.5
42.0
42.5
 (degrees)
0
2
4
6
8
Count
 distribution
mean=40.479
median=40.365
(a) ∆θ distribution.
0.220 0.225 0.230 0.235 0.240 0.245 0.250 0.255 0.260
alignment gain (cosine)
0
2
4
6
8
Count
Alignment gain: vimg + txt, vsteer
vtext, vsteer
mean=0.239
median=0.238
(b) Alignment gain histogram.
0
1
-cluster
39.0
39.5
40.0
40.5
41.0
41.5
42.0
 (degrees)
[
Ck] (per cluster)
(c) E[∆θ | Ck] by
cluster.
0
1
-cluster
0.220
0.225
0.230
0.235
0.240
0.245
0.250
0.255
0.260
alignment gain (cosine)
Alignment gain by cluster
(d) Alignment gain
by cluster.
0
1
Cluster
0.70
0.75
0.80
0.85
0.90
avg_gate
avg_gate by cluster (Beta space)
(e) gate by cluster
(sanity).
Figure 6. Directional evidence with reflowed layout. (a) Consis-
tent vtext →vimg+txt rotation; (b) positive alignment gain to vsteer;
(c,d) cluster-wise stability; (e) systematic gate differences.
B. Additional Experiment
B.1. LLaVA Internal Dynamic Analysis
As mentioned in Section 3.2, our method is guided by an
analysis of the internal dynamics of LLaVA-1.5 (Liu et al.,
2024a), with key findings illustrated in Figure 9. By examin-
ing the residual update vector from the self-attention module
at each layer, we identified two properties that informed our
intervention strategy:
Intervention Leverage Peaks in Late Layers. We can find
that the magnitude of the residual update, which represents
the “leverage” an intervention can have, is not uniform. We
found that its strength grows with model depth, peaking
in the late decoder blocks (approx. layers 26-32). This
indicates that interventions in these layers have the greatest
potential to influence the model’s final output (Figure 9a,
9c).
Directional Coherence Collapses in Late Layers. While
late layers offer the most leverage, the directional coherence
of their update vectors collapses after approximately layer 21
(Figure 9b). Coherence is moderate only in the early-to-mid
layers. This suggests that applying a fixed, global steering
vector in the high-leverage late layers is suboptimal, as the
intervention may be misaligned with the model’s unstable
internal state.
B.2. Additional Ablation Results
We provide supplementary abalation results on LLaVA-1.5
and InstructBLIP, as shown in Figure 15 and Figure 16.
These analyses complement the main ablation study con-
ducted on Idefics2 in Section 4.4.
We further verify whether the CHAIR-tuned hyperparame-
12

## Page 13

Adaptive Residual-Update Steering for Hallucination Mitigation
Figure 7. CARD vectors cluster by visual semantic content. We compute CARD vectors from LLaVA-1.5-7B on four COCO
categories with 50 images per category and visualize them using t-SNE. The category-level clustering indicates that CARD captures
image-conditioned visual semantics rather than random activation noise.
ters transfer to POPE. Figure 8 shows that all tested config-
urations improve over the vanilla baseline, and the CHAIR-
selected setting remains close to the best POPE configura-
tion. This supports the use of a small held-out calibration
set rather than benchmark-specific retuning.
The results for both models confirm the same core trends
observed with Idefics2. Specifically, the heatmaps reveal a
consistent trade-off between hallucination mitigation and re-
call. As the steering strength αmax increases, both CHAIRS
and CHAIRI scores improve (decrease), but this is often
accompanied by a drop in recall. The gate sensitivity pa-
rameter, k, plays a similar, non-linear modulating role in
this balance. While the general trade-off is consistent, the
optimal hyperparameter values vary by model architecture,
highlighting the need for model-specific tuning.
B.3. Case Study
To provide a qualitative illustration of our method’s real-
world performance, we present a series of case studies in
Figures 12, 13 and 14. From these examples, we can find
that RUDDER is highly effective at eliminating object-level
hallucinations. It successfully removes entirely non-existent
objects from the captions (e.g., a hallucinated “second per-
son” or “cup”), and corrects misidentified objects (e.g., cor-
rectly identifying “skis” instead of “snowboard”). Moreover,
RUDDER’s outputs are not only more factually accurate
but also more semantically cautious. The corrected descrip-
tions often adopt more conservative language, such as using
phrases like “appears to be”, “may be”, or “suggesting that”.
By replacing the baseline models’ confident yet incorrect
assertions with more grounded and appropriately qualified
statements, RUDDER significantly enhances the overall
reliability and trustworthiness of the generated text.
To further inspect the behavior of the adaptive gate, we visu-
alize token-level gate values and CARD similarities for two
representative samples. Figures 10 and 11 show that visually
grounded content tokens generally receive stronger align-
ment and gate values, whereas tokens associated with un-
supported concepts receive lower CARD similarity. These
examples complement the caption-level case studies by
13

## Page 14

Adaptive Residual-Update Steering for Hallucination Mitigation
Figure 8. POPE hyperparameter sensitivity on LLaVA-1.5-7B. All tested configurations improve over the baseline, with a small spread
across the grid. The CHAIR-tuned setting remains close to the best POPE configuration, suggesting that RUDDER is not highly sensitive
to the exact choice of αmax and k.
showing that the Beta Gate operates at the intended token
granularity.
C. Clarifications Added After Reviewer
Discussion
Bayesian-inspired rather than full Bayesian inference.
Our Beta Gate is not intended to perform full posterior in-
ference over model parameters or generated tokens. Instead,
it uses the Beta–Bernoulli posterior mean as a structured,
bounded, and monotone mapping from visual-alignment
evidence to an intervention strength. The softplus pseudo-
counts ensure smoothness and positivity, the concentration
parameter c controls the prior strength, and the clamp range
provides safety rails against degenerate gates.
Why reinforce high-alignment states.
The gate should
be interpreted as a trust estimator for the current visual
trajectory, not as an error magnitude estimator. When st =
cos(hl,t, vCARD) is high, the hidden state is already aligned
with the visual evidence extracted from prefill, so adding the
CARD direction reinforces grounded generation. When st
is low, the token may be syntactic or the representation may
be unstable; reducing the intervention avoids over-steering
and helps preserve fluency and recall.
Layer selection across architectures.
The best interven-
tion layer is architecture-dependent. For LLaVA-style mod-
els and Idefics2, visual tokens are concatenated or projected
into the LLM input, and the visual signal tends to fade
in middle-to-late language layers; hence late layers offer
stronger leverage. For Q-Former-based or stronger early-
fusion models such as InstructBLIP and Qwen2.5-VL, use-
ful multimodal alignment can occur earlier, so early inter-
vention can be preferable. We therefore use a small held-out
sweep to select a fixed layer per backbone and then keep
that layer unchanged across benchmarks.
Efficiency claim.
“No extra forward pass” means that
RUDDER does not invoke an additional model forward be-
yond the vanilla prefill and autoregressive decoding calls.
CARD is extracted from the mandatory prefill pass through
a read-only hook, while decoding only adds cheap vector
operations and a residual-stream update. This is why RUD-
DER keeps the end-to-end latency overhead below 4% in
our measurements.
Hallucination–recall trade-off.
A method can trivially
reduce hallucination by becoming overly conservative. To
avoid this failure mode, our calibration explicitly requires
at least 95% of baseline recall, and we additionally evalu-
ate general multimodal capability on MME. This ensures
that RUDDER reduces unsupported object mentions while
largely preserving correct object recall and general visual
reasoning behavior.
D. LLMs Usage Statement
Generative AI has been utilized to enhance the writing and
to assist with coding tasks.
14

## Page 15

Adaptive Residual-Update Steering for Hallucination Mitigation
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
Layer index
0.0
2.5
5.0
7.5
10.0
12.5
15.0
17.5
Mean token L2( )
0.810
1.603
2.884
2.020
2.505
3.138
5.340
4.451
5.566
5.510
6.934
6.888
5.558
6.024
6.457
5.610
5.073
5.021
4.641
5.633
5.404
4.265
7.050
5.005
8.788
5.130
10.110
6.824
8.770
11.432
12.169
18.488
Self-Attn  strength per layer
(a) Absolute strength of the update vector (∥δl∥) across layers.
0
5
10
15
20
25
30
Layer index
0.0
0.2
0.4
0.6
0.8
1.0
Mean cosine to group mean
CARD direction coherence per layer (to group mean)
(b) Directional coherence within layer groups.
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
Layer index
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Mean L2( )/L2(pre)
0.208
0.216
0.241
0.108
0.142
0.173
0.240
0.191
0.239
0.232
0.287
0.261
0.210
0.226
0.235
0.204
0.181
0.177
0.156
0.193
0.187
0.140
0.227
0.158
0.278
0.160
0.321
0.224
0.288
0.389
0.415
0.766
Relative change of Self-Attn (  vs pre) per layer
(c) Relative strength of the update vector (∥δl∥/∥hl
pre∥).
Figure 9. Analysis of the internal dynamics of LLaVA-1.5 (Liu
et al., 2024a). (a, c) Both the absolute and relative strength of self-
attention updates peak in the middle-to-late layers, which identify
a ”computational core”. (b) Mid-late layers show significantly
lower directional coherence than other regions.
15

## Page 16

Adaptive Residual-Update Steering for Hallucination Mitigation
(a) Input image.
(b) Token-level gate and CARD similarity.
Figure 10. Token-level Beta Gate visualization for a skiing image. Visually grounded tokens receive consistently high gate values,
while less visually grounded or more abstract tokens show lower CARD similarity. This supports the interpretation of the gate as a
visual-grounding-dependent trust signal.
(a) Input image.
(b) Token-level gate and CARD similarity.
Figure 11. Token-level Beta Gate visualization for a cell-phone image. The hallucinated token sequence related to “remote control”
receives low CARD similarity, showing that the gate can down-weight intervention when the current hidden state is poorly aligned with
the image-conditioned CARD direction.
16

## Page 17

Adaptive Residual-Update Steering for Hallucination Mitigation
Baseline: 
“The image features a small, clean kitchen with a white countertop and a 
stove top oven. The kitchen is equipped with a sink and a microwave, and 
there is a toilet visible in the background. The room also has a dining table 
and a few chairs placed around it. In addition to the main kitchen appliances, 
there are a few other items in the room. "
RUDDER: 
"The image features a clean and tidy kitchen with a variety of appliances 
and utensils. There is a sink positioned towards the center of the room, with 
a microwave oven placed above it. A refrigerator can be seen on the left 
side of the room, while an oven is located towards the bottom left corner.
Baseline: 
"The image features a cozy living room with a couch and a chair. On the 
couch, there are two cats lying down, one on the left side and the other on 
the right side. They appear to be relaxed and enjoying their time together. In 
addition to the cats, there are two books placed on the couch, one near the 
left cat and the other near the right cat. The scene gives off a warm and 
comfortable atmosphere, with the cats and books adding to the overall 
ambiance of the living room."
RUDDER: 
"The image features two cats lying down on a couch, enjoying each other' 
sierp company. One cat is positioned towards the left side of the couch, 
while the other cat is laying down further to the right side. They both seem 
to be resting comfortably on newspapers that are spread out on the couch. 
The scene creates a cozy atmosphere with the cats sharing their space on 
the couch."
Baseline: 
"The image features a wooden table with a variety of kitchen utensils and 
items. There are several knives, including a large knife and a smaller one, 
placed on the table. A spoon is also visible, along with a couple of spoons and 
a fork. In addition to the utensils, there are multiple bottles on the table, 
with one being larger and positioned towards the left side, and the others 
being smaller and scattered around the table. A cup can be seen on the right 
side of the table, and a bowl is placed in the middle. table."
RUDDER: 
“The image features a tabletop with a variety of items laid out on it. There 
are several spices neatly arranged in a line. Additionally, a number of spoons 
are positioned around the tabletop, some pointing upwards and others 
pointing downwards. There are also several bottles dispersed across the 
tabletop, adding to the assortment of items present. The scene gives off an 
impression of a well-organized and functional space for preparing meals or 
engaging in culinary activities."
Baseline: 
"The image features a red fire hydrant situated on a sidewalk next to a 
building. The fire hydrant is prominently placed in the scene, and it appears to 
be the main focus of the image. The sidewalk is located near a street, and 
there is a truck parked on the street, slightly to the left of the fire hydrant."
RUDDER: 
“The image features a red fire hydrant prominently positioned on the 
sidewalk near a building wall. The fire hydrant is sitting on top of a sidewalk 
curbing, making it easily accessible for emergency services. In addition to the 
fire hydrant, there are several graffiti tags on the building wall behind it. 
These tags add an urban touch to the scene and provide contrasting visual 
interest against the red fire hydrant."
Figure 12. Case study. Hallucinated contents generated by the vanilla LLaVA-1.5 are marked in red, while the factual contents from our
method are colored with blue.
17

## Page 18

Adaptive Residual-Update Steering for Hallucination Mitigation
Baseline: 
"The image features a man wearing a jacket and a cap, standing on a snowy 
hill. He appears to be posing for a photo while holding his snowboard. The 
man has his hands close to his chest. The snowy terrain surrounds the man, 
and in the background, there are additional trees scattered across the area. A 
second person is visible in the background, standing further to the right. "
RUDDER: 
"The image features a man posing on top of a snow-covered ski slope during 
winter. He is wearing snow gear and is standing confidently on skis. The 
scene appears to be captured during the winter season, with the 
mountainous setting serving as a striking background."
Baseline: 
“The image features a busy city street with several police officers patrolling 
the area on their motorcycles. There are three police officers prominently 
visible in the scene, with one officer positioned slightly to the left of the 
center, another officer slightly to the right of the center, and the third officer 
slightly to the right of the rightmost officer. Multiple cars are parked or 
driving down the street, including one car parked on the left side of the 
image, another car parked further down the street on the right side, and 
several cars driving down the roadway. Additionally, there are two handbags 
visible in the scene one near the leftmost officer and the other slightly to 
the right of the center officer."
RUDDER: 
“The image depicts a busy city street filled with various vehicles and 
pedestrians. There are two police officers on motorcycles, one on the left 
side and the other on the right side of the street. They appear to be 
directing traffic or monitoring the area. In addition to the motorcycles, 
there are several cars and a truck on the street. Some cars are parked, while 
others seem to be in motion. A few pedestrians can be seen walking along 
the sidewalk, with one person carrying a handbag. Overall, the scene 
captures the hustle and bustle of city life, with the police officers playing a 
crucial role in maintaining order and safety."
Baseline: 
" The image captures a lively concert scene with a crowd of people gathered 
around a stage. A man is standing on the stage, singing into a microphone, 
while a woman is also present on the stage, possibly accompanying the singer.
There are several people in the audience, some of them holding cell phones, 
likely capturing the moment or sharing the experience with others. The 
atmosphere appears to be energetic and engaging, with the performers and 
the audience enjoying the concert together."
RUDDER: 
“The image features a live concert event taking place indoors. At the center
of attention is a person singing into a microphone while a band plays 
instruments behind them. There are several other people watching the 
performance from various positions around the room. Some of these 
individuals are standing close to each other, while others are spread out 
across the venue. In addition to the live performance setup, there are 
multiple cell phones being used by concertgoers to capture memories of the 
event. ”
Baseline: 
"The image features a black cat sitting in a bathroom sink, surrounded by 
various toiletries. The sink is filled with toothbrushes, some of which are 
placed near the cat, while others are scattered around the sink. There are also 
multiple bottles in the scene, with some located near the cat and others 
placed further away. The cat appears to be relaxed and comfortable in the 
sink, surrounded by the bathroom items."
RUDDER: 
"The image features a black cat comfortably lying down inside a bathroom 
sink. The sink occupies most of the scene, with the cat occupying its central 
space. The cat appears to be enjoying its time in the sink, possibly finding it 
cozy or cooler than its surroundings."
Figure 13. Case study. Hallucinated contents generated by the vanilla Idefics2 are marked in red, while the factual contents from our
method are colored with blue.
18

## Page 19

Adaptive Residual-Update Steering for Hallucination Mitigation
Baseline: 
" There are four plates with different types of cakes and pastries, each with a 
unique flavor and design. The desserts are arranged in a way that they are 
easily accessible for the diners. The table is set with utensils, including a fork 
and a knife, and a cup is also present. The dining area is well-lit, providing a 
pleasant atmosphere for enjoying the delicious treats."
RUDDER: 
" There are four plates on the table, each containing a different dessert. The 
dessert selection includes a slice of cake, a slice of cheesecake, a slice of 
lemon cake, and a slice of chocolate cake. The dessert plates are arranged in 
a way that allows for a variety of flavors to be enjoyed."
Baseline: 
" He is looking upwards, possibly in a contemplative or thoughtful manner. 
The room appears to be a dining area or a gathering place, with a table and 
chairs visible in the background. The man's facial expression and body 
posture suggest a sense of seriousness or concentration "
RUDDER: 
"He is looking upwards, possibly at a person or object in the room. The man 
is the focal point of the scene, and his presence is quite prominent. There 
are two lights in the room, one on the left side and another on the right 
side. The man is the center of the scene, and his presence is quite 
prominent."
Baseline: 
" Two men and a woman present, all wearing different clothing. One man is 
holding a kite, while the other man is holding a bottle. The woman is standing 
nearby, possibly assisting with the kite or enjoying the activity. "
RUDDER: 
" There are two kites in the air, one of a kite shaped like a killer whale. The 
people are having fun and flying the kites, possibly enjoying the day 
together."
Baseline: 
" The baby is wearing a white outfit, and the man is smiling at the camera. 
The baby is positioned close to the man's face, making it a heartwarming
moment captured in the photo. There are several other people in the 
background, including a man and a woman, both standing in the same room. 
The man and woman are positioned on opposite sides of the room, with the 
man holding the baby and the woman standing in the background "
RUDDER: 
" The baby is wearing a white outfit, and the man is smiling at the camera. 
There are two other people in the room, one on the left side and another on 
the right side. The room appears to be a church, with a cross visible in the 
background. The man and the baby are the focal point of the scene."
Figure 14. Case study. Hallucinated contents generated by the vanilla InstructBLIP are marked in red, while the factual contents from our
method are colored with blue.
19

## Page 20

Adaptive Residual-Update Steering for Hallucination Mitigation
0.0
5.0
10.0
15.0
20.0
25.0
30.0
Strength 
max
0.0
5.0
10.0
15.0
20.0
Sensitivity K
48.4
46.6
43.4
44.0
44.2
43.8
40.0
48.4
45.6
43.8
44.6
38.0
27.4
28.0
48.4
44.8
42.8
44.8
32.0
26.8
28.0
48.4
45.0
42.4
43.2
30.8
29.8
30.4
48.4
45.2
44.0
40.2
32.0
31.6
27.8
CHAIRs (
) 
 layer 30  (value: %)
27.5
30.0
32.5
35.0
37.5
40.0
42.5
45.0
47.5
Value (%)
(a) CHAIRS αmax × k heatmap.
0.0
5.0
10.0
15.0
20.0
25.0
30.0
Strength 
max
0.0
5.0
10.0
15.0
20.0
Sensitivity K
13.9
13.5
12.9
12.5
13.8
13.8
13.3
13.9
13.6
12.6
12.2
10.5
11.1
10.8
13.9
13.4
12.8
12.6
11.4
10.1
10.5
13.9
13.5
12.6
12.2
11.6
11.1
11.2
13.9
13.8
12.8
11.3
11.4
10.9
10.2
CHAIRi (
) 
 layer 30  (value: %)
10.5
11.0
11.5
12.0
12.5
13.0
13.5
Value (%)
(b) CHAIRi αmax × k heatmap.
0.0
5.0
10.0
15.0
20.0
25.0
30.0
Strength 
max
0.0
5.0
10.0
15.0
20.0
Sensitivity K
72.9
72.7
71.8
72.0
72.0
71.5
68.9
72.9
72.2
72.1
71.4
69.4
59.1
58.8
72.9
71.5
71.9
70.0
61.8
60.0
60.4
72.9
71.5
71.6
69.1
61.5
60.9
60.4
72.9
71.5
71.9
69.1
62.1
61.8
59.2
Recall ( ) 
 layer 30  (value: %)
60
62
64
66
68
70
72
Value (%)
(c) Recall αmax × k heatmap.
Figure 15. Ablation matrices for RUDDER on LLaVA-1.5 (Liu et al., 2024a)
0.0
2.5
5.0
5.5
6.0
6.5
7.0
Strength 
max
0.0
2.0
4.0
6.0
8.0
Sensitivity K
39.2
36.2
32.8
35.0
29.8
18.2
10.2
39.2
37.2
35.2
34.6
30.4
20.6
13.6
39.2
36.0
35.0
35.4
31.6
24.4
14.0
39.2
36.8
35.0
34.2
34.2
25.6
16.0
39.2
36.2
34.2
35.6
33.2
27.6
17.6
CHAIRs (
) 
 layer 1  (value: %)
15
20
25
30
35
Value (%)
(a) CHAIRS αmax × k heatmap.
0.0
2.5
5.0
5.5
6.0
6.5
7.0
Strength 
max
0.0
2.0
4.0
6.0
8.0
Sensitivity K
12.6
11.5
9.9
11.9
9.0
6.2
3.7
12.6
11.9
11.0
10.3
9.1
7.2
6.6
12.6
10.3
11.2
11.7
10.3
9.7
6.8
12.6
11.8
11.5
10.9
11.3
10.3
6.3
12.6
10.6
10.6
11.5
10.2
8.5
7.0
CHAIRi (
) 
 layer 1  (value: %)
4
5
6
7
8
9
10
11
12
Value (%)
(b) CHAIRi αmax × k heatmap.
0.0
2.5
5.0
5.5
6.0
6.5
7.0
Strength 
max
0.0
2.0
4.0
6.0
8.0
Sensitivity K
62.0
63.1
63.0
62.5
60.2
53.9
47.5
62.0
62.8
63.4
62.8
60.3
55.5
48.7
62.0
62.8
61.6
62.4
61.7
57.6
51.5
62.0
63.5
61.8
62.7
62.0
57.2
53.1
62.0
63.5
62.7
62.4
60.8
59.9
53.9
Recall ( ) 
 layer 1  (value: %)
48
50
52
54
56
58
60
62
Value (%)
(c) Recall αmax × k heatmap.
Figure 16. Ablation matrices for RUDDER on InstructBLIP (Dai et al., 2023)
20
