---
source_pdf: papers/GASP - Detecting Hallucinations in Retrieval-Augmented Generation through Grounding-Aware Sensitivity by Perturbation.pdf
slug: gasp-detecting-hallucinations-in-retrieval-augmented-generat
pages: 23
extracted_on: 2026-08-09
---

# GASP - Detecting Hallucinations in Retrieval-Augmented Generation through Grounding-Aware Sensitivity by Perturbation

## Page 1

Research Article
Bouke, 2026
Detecting Hallucinations in Retrieval-Augmented Generation
through Grounding-Aware Sensitivity by Perturbation (GASP)
Mohamed Aly Bouke
#1,*
1Centre for Intelligent Cloud Computing, CoE for Advanced Cloud, Faculty of Information Science and Technology,
Multimedia University, Jalan Ayer Keroh Lama, Bukit Beruang, 75450, Melaka, Malaysia
*bouke@ieee.org, alybouke@mmu.edu.my
Research Article, July 7, 2026
Abstract
Retrieval-augmented generation (RAG) reduces but does not eliminate hallucination, and existing detectors return a
single answer-level score that does not indicate which sentence is unsupported, or why. To close this gap, we introduce
Grounding-Aware Sensitivity by Perturbation (GASP), a span-level detector that scores each answer sentence by how
strongly its likelihood depends on the retrieved evidence, a quantity we term grounding sensitivity. GASP holds the
answer fixed and re-scores it under the full context, under no context, and with each chunk removed, then measures
the log-likelihood drops and Jensen-Shannon divergences (JSD). The likelihood of a grounded sentence collapses
once its supporting passage is removed, whereas a hallucinated sentence is almost unaffected, a contrast we interpret
by casting decoding as a random nonlinear iterated function system (RNIFS). We evaluate GASP on three benchmarks
(RAGTruth, TofuEval, RAGBench) with three instruction-tuned scorers from two model families (Qwen2.5-0.5B,
Qwen2.5-1.5B, and SmolLM2-1.7B) under a leakage-clean protocol. On RAGTruth it reaches a response-level area
under the receiver operating characteristic (ROC) curve (AUC) of about 0.73 and a span-level AUC of about 0.67,
improving significantly over perplexity and by clear margins over length, whole-context natural language inference
(NLI), and self-consistency baselines. The only baseline competitive at the span level is a well-configured chunk-level
entailment verifier, which requires a separate model, whereas a training-free threshold on the grounding features
matches the trained classifier without labeled data and serves as the default detector. Beyond RAGTruth, the signal
transfers to TofuEval but not to short-answer question answering in RAGBench, showing GASP is best suited to
outputs constructed from the retrieved context rather than answers recoverable from parametric knowledge.
Keywords: hallucination detection, retrieval-augmented generation, context perturbation, explainable AI, large language models
1
Introduction
Large language models (LLMs) are increasingly deployed with
RAG, where a retriever supplies passages that the model condi-
tions on when answering a query [1]. Grounding generation in
retrieved evidence narrows the gap between what a model as-
serts and what can be verified, and it has become the dominant
pattern for question answering, summarization, and assistant
systems in knowledge-intensive settings. Retrieval injects up-
to-date and domain-specific information that a fixed parametric
model cannot hold, and it offers a route to attribution, since
each claim can in principle be traced to a retrieved passage. The
pattern is now spreading into settings where the cost of an error
is high, such as clinical decision support, legal research, and
financial analysis, which raises the stakes of any unsupported
statement the system produces and calls for checking grounding
at the level of the individual claim.
Yet retrieval does not remove the core failure of generative
models. They continue to produce fluent statements that the
supplied evidence does not support, a phenomenon usually
called hallucination [2], [3]. The model remains free to blend
its parametric prior with the retrieved text, to over-generalize
from a partial match, or to contradict a passage it has nominally
read. A single unsupported sentence that reads as confidently as
a true one can cause real harm. Consider a clinical assistant that
retrieves passages from a drug formulary and answers a dosing
question. If one sentence states a dose that no retrieved passage
supports, the answer as a whole may still read as authoritative,
and a busy clinician may not catch the single unsupported clause.
The practical question is therefore not only whether an answer
is trustworthy as a whole, but which specific parts of it are
unsupported, and why.
Two limitations constrain hallucination detectors in practice.
The first is granularity and transparency. Most detectors return
one score for an entire answer, so a reviewer learns that the
answer might be unreliable but not which sentence is unsup-
ported, nor why it was flagged. A response-level score is poorly
matched to how people verify text, since verification proceeds
claim by claim. The second limitation concerns the signal it-
self. A natural first attempt is to read intrinsic properties of the
generated text, such as its perplexity or token-level uncertainty,
and to treat unusual values as evidence of hallucination. The
difficulty is that an unsupported sentence can be as fluent and
internally ordinary as a true one, so intrinsic signals measure
how the model writes, not whether what it writes is supported.
The central observation of this work is that hallucination is
a property of the relationship between a sentence and its ev-
idence, not of the sentence in isolation. A faithful sentence
is constructed from a particular passage and depends on it,
so its plausibility to the model collapses if that passage is re-
moved, whereas a hallucinated sentence is produced from the
Page 1 of 23
arXiv:2607.04223v1  [cs.CL]  5 Jul 2026

## Page 2

Research Article
Bouke, 2026
model prior and is unchanged by the removal. We turn this
into a measurable signal by perturbing the retrieved context
and observing how much each answer span reacts. We give the
signal a dynamical-systems reading by viewing decoding under
a context as an RNIFS whose attractor is the set of grounded
continuations, so that grounding sensitivity is how strongly that
attractor reacts to perturbations of the evidence that defines
it. This reading connects the detector to the theory of how an
invariant measure responds to perturbation and clarifies why a
measure-theoretic distance, rather than a surface text statistic,
is the right instrument.
A practical consequence of testing grounding by perturbation
is that the detector is cheap and self-contained. It needs only
the ability to score a fixed answer under a few variants of the
context, which a small model provides, so it adds no separate
verifier to maintain and no resampling of the answer. The one
requirement is access to a probabilistic scorer that can return
token likelihoods under a given context. This scorer can be
a different and smaller model than the one that produced the
answer, which lets the same guardrail audit a large or hosted
generator whose internals are unavailable, though it does not ap-
ply to a generator that exposes neither its own log-probabilities
nor a stand-in scorer.
The objective of this paper is to design and evaluate a span-
level RAG hallucination detector that scores each answer sen-
tence by its grounding sensitivity and that returns, for each
flag, the context unit that best supports the span as an explana-
tion. We require the detector to outperform perplexity, length,
and whole-context entailment, and to be at least competitive
with a well-configured chunk-level entailment verifier, under a
leakage-clean evaluation that prevents sentences from the same
answer from appearing in both training and test folds. The
intended use is an explainable guardrail and audit tool for RAG
assistants, where a reviewer or an automated filter sees, for each
sentence, whether it is anchored in the retrieved evidence, sees
the supporting passage when it is, and receives an alert when a
sentence is anchored in nothing.
The main contributions of this paper are as follows.
1. We develop GASP, a span-level RAG hallucination detec-
tor that scores each sentence by its grounding sensitivity,
the change in the sentence likelihood when the retrieved
context is perturbed.
2. We provide a training-free threshold-only variant that
requires no labeled data.
3. We provide an attribution mechanism that returns a can-
didate supporting passage for each sentence at no extra
cost, and evaluate it against NLI and lexical controls.
4. We evaluate the detector under a leakage-clean protocol
across three benchmarks (RAGTruth, TofuEval, RAG-
Bench) and three scorers from two families, with con-
fidence intervals, significance tests, and ablations, and
map where the signal does and does not transfer.
5. As a supporting interpretation, we cast decoding as an
RNIFS and bound the movement of its invariant measure
by the divergence the detector reads, testing this on the
hidden states.
The remainder of this paper is organized as follows. Section 2
reviews related work and positions the contribution. Section 3
presents preliminaries. Section 4 formalizes the task. Section 5
presents the method. Section 6 describes the experimental setup,
Section 7 reports the detection results, and Section 8 presents
the analysis and diagnostics. Section 9 discusses interpretation
and use, Section 10 examines threats to validity, and Section 11
considers ethical use. Section 12 concludes, and Section 13
discusses limitations and future work.
2 Related Work
2.1 Intrinsic and uncertainty-based detection
A first family of detectors reads signals from the model itself,
including token probability, perplexity, and uncertainty esti-
mates such as semantic entropy that aggregate over meaning-
equivalent generations [4]. A recent line reads the model in-
ternal states, since hidden activations appear to retain informa-
tion about whether an output is hallucinated [5], and related
work mines attention and probability statistics for the same
purpose [6]. These methods require no external resources and
run during or after generation. Their weakness is that fluency
and confidence are weak proxies for grounding, since a model
can be fluent and confident about an unsupported statement and
uncertain about a correct but rare phrasing. Calibration com-
pounds the problem, because instruction tuning and decoding
choices shift the probabilities, so a fixed threshold transfers
badly across models and prompts. These signals describe the
model state rather than the evidence, which leaves a gap for
detectors that test grounding directly and at the granularity of
an individual claim. Uncertainty methods that aggregate over
meaning-equivalent samples mitigate part of the calibration is-
sue, but at the cost of several generations per answer, and they
report a quantity that is still about the distribution of model
outputs rather than about the relationship between an output
and the retrieved evidence. The persistent gap is a signal that is
both grounding-aware and localized.
2.2 Sampling-consistency detection
A second family samples several generations for the same
prompt and treats disagreement among them as evidence of
hallucination, as in SelfCheckGPT [7]. Consistency is infor-
mative when a model is internally unsure, and the approach is
black-box and needs no labeled data. The signal is computed,
however, by perturbing the output through resampling rather
than by perturbing the grounding, and it needs several gener-
ations per answer, which multiplies the inference budget. A
model can be confidently and consistently wrong, repeating
the same unsupported claim across samples, so the unanswered
question remains whether the answer depends on the supplied
evidence. The agreement is also usually scored by surface com-
parison at the response or sentence level, which couples the
detector to a text-similarity model and inherits its blind spots,
and it does not by itself indicate which passage should have
supported a flagged sentence. The resampling view and the
context-perturbation view are in fact complementary, since one
probes the stability of the output under the model own stochas-
ticity while the other probes the dependence of the output on
the external evidence, and the two answer different questions.
Page 2 of 23

## Page 3

Research Article
Bouke, 2026
A complete trustworthy generation stack might use both, but for
the specific question of whether an answer is grounded in the
retrieved passages, perturbing the grounding is the more direct
test, and it is the one that also yields the supporting passage as
a byproduct.
2.3 Entailment and fact verification
A third family checks whether the answer is entailed by the con-
text using NLI models or fact-verification pipelines, including
fine-grained scoring that decomposes a response into atomic
facts and verifies each [8]. These methods target grounding
directly, which is the right objective, but they depend on a sep-
arate verification model whose domain and granularity may
not match the target setting, and entailment over long, multi-
sentence context is brittle. Their effectiveness is bounded by
the quality and transfer of the auxiliary model, which leaves
room for a signal computed by the scoring model itself without
a separate verifier. The objectives also diverge in deployment,
since an external verifier must be maintained, versioned, and
matched to the domain of the system it audits, whereas a signal
derived from the generator scales with the system and requires
no second model to keep current. Decomposition into atomic
facts improves localization but multiplies the number of verifier
calls, which is costly to run on every answer in production.
There is also a representational mismatch, since an entailment
model trained on short premise and hypothesis pairs is asked
here to judge a long multi-passage premise against a sentence
drawn from a longer answer, and its calibration on this out-
of-distribution input is uncertain. Retrieval-grounded answers
frequently paraphrase, aggregate, or partially restate several
passages, and an off-the-shelf verifier may score such faithful
aggregation as only weakly entailed while scoring a fluent but
unsupported sentence on the same topic as plausibly entailed,
which is the opposite of what a grounding detector needs. These
difficulties are not arguments against entailment in principle,
but they explain why a grounding signal read from the genera-
tor, which does not project the problem onto a separate model
trained for a different task, complements it.
2.4 Context attribution and ablation
A fourth family ablates context sources to attribute a generation
to the passages that produced it, as in ContextCite [9], and a
broader methodological thread analyzes models by leave-one-
out input ablation and counterfactual edits. A closely related
recent method reads attention from the answer back to the con-
text and flags a contextual hallucination when too little attention
falls on the retrieved passages [10], but attention indicates where
the model looks rather than whether the output would change
if a passage were gone, so it is a correlational rather than a
counterfactual signal. A further mechanistic line pursues the
same context-versus-knowledge question by reading the gener-
ator’s internal state directly. ReDeEP attributes a hallucination
to knowledge feed-forward modules that overweight parametric
memory and to copying attention heads that fail to carry the
retrieved evidence forward [11], and LUMINA pairs a distribu-
tional measure of external-context use with a layer-wise trace of
how the predicted token evolves inside the model [12]. These
signals are informative, yet they are read from the model’s
activations, so they require white-box access to the generator
and describe the mechanism correlationally rather than testing
dependence directly. The ablation strand, by contrast, estab-
lishes that responses depend on identifiable sources and that
removing them changes the output, which is the causal lever
we use. It is framed as attribution, however, with the goal of
crediting sources for a generation assumed worth citing, rather
than as a detector that uses the absence of such dependence
to flag content as unsupported. Attribution is content with a
ranking of sources, whereas detection must decide whether any
passage supports a span at all and convert the ablation effect
into a calibrated decision across spans, so a method tuned for
the first is not automatically a detector for the second. The de-
tection reading also calls for a principled readout of the ablation
effect. A change in a single realized token probability is noisy,
whereas a divergence between the full predictive distributions is
more stable and admits the measure-theoretic interpretation we
develop, in which the ablation moves a context-conditioned in-
variant measure by an amount that reflects the span dependence
on the removed unit.
2.5 Faithfulness and dynamical systems
Work on summarization faithfulness establishes that faithfulness
is distinct from fluency and is best assessed at a fine granularity,
and it provides span-annotated corpora and atomic-fact proto-
cols [8], [13]. These metrics are typically computed by an exter-
nal judge and are designed to score a finished summary rather
than to provide a cheap, self-contained signal, a pattern contin-
ued by automated retrieval-augmented evaluation suites [14]
and by recent retrieval-augmented hallucination detectors that
train a dedicated judging model [15]. Separately, work on
generative and symbolic processes as dynamical systems char-
acterizes their attractors and stability, including RNIFS and
their dimensional properties [16]. This perspective supplies a
vocabulary of invariant measures and stability under perturba-
tion that has not been connected to the practical problem of
deciding whether a generated span is grounded.
2.6 Positioning
Across these threads a single need emerges. Intrinsic and sam-
pling methods do not test grounding, entailment methods import
a separate and brittle verifier, context-ablation methods estab-
lish dependence but read it as attribution rather than detection,
mechanistic methods infer context reliance from the genera-
tor’s internal state and so require white-box access to it, and
the dynamical-systems perspective offers stability tools that
have not been applied to grounding. The present work sits at
this intersection. We use context perturbation as a detection
signal, we compute it from a probabilistic scorer, which need
not be the model that produced the answer, without a separate
trained verifier, and we interpret it through the stability of a
context-conditioned invariant measure. The contribution is not
a new model but a new reading of an existing causal lever,
since context ablation is already known to change a generation,
and the present work shows that the magnitude of that change,
measured as a movement of a context-conditioned invariant
measure and aggregated to the span, is a usable and explain-
able hallucination signal that needs no auxiliary verifier and no
resampling. Table 1 places grounding sensitivity among these
families, where it is set apart by reading a counterfactual con-
Page 3 of 23

## Page 4

Research Article
Bouke, 2026
Table 1: Qualitative comparison of hallucination-detection families along the axes that matter for grounded, explainable detection, with a
representative method for each family.
Family
Signal source
What is perturbed
Granularity
Separate
verifier
Built-in
explanation
Example methods
Intrinsic
/
uncer-
tainty
token probability, en-
tropy
nothing
response or token
no
no
semantic entropy, IN-
SIDE [4], [5]
Sampling
consis-
tency
agreement of sam-
ples
the output (resam-
pling)
response or sen-
tence
no
no
SelfCheckGPT [7]
Entailment
/
fact
check
external NLI or veri-
fier
nothing
sentence or fact
yes
partial
FActScore [8]
Trained detector
supervised classifier
nothing
span or token
yes
partial
LettuceDetect [15]
Context attribution
change under abla-
tion
the
context
(sources)
source attribution
no
yes
ContextCite [9]
Mechanistic / inter-
nal state
attention, FFN acti-
vations
nothing
response or token
no
partial
Lookback Lens,
Re-
DeEP, LUMINA [10],
[11], [12]
GASP
change under abla-
tion
the
context
(chunks)
response and span
no
yes
this work
text perturbation at the span level and returning the supporting
passage with its decision, rather than relying on an intrinsic
score, a resampling test, or a separate verifier.
3
Preliminaries
3.1 RAG and hallucination
In RAG, a query q is used to retrieve a set of context passages
c, and an answer y is generated by an autoregressive model
conditioned on q and c. The intent is that every claim in y
should be entailed by c, and hallucination is the violation of this
intent. It is useful to separate two notions that are easily con-
flated. Factuality asks whether a statement is true in the world,
which requires a trusted external source to judge, while faith-
fulness, or grounding, asks whether a statement is supported
by the specific evidence the system was given. In RAG the
operative contract is grounding, since the system promises to
answer from the retrieved passages, so grounding is our target.
Concretely, the benchmark we use separates two grounding fail-
ures whose relation to the context differs. Baseless information
introduces details absent from the evidence, and evident conflict
contradicts the evidence. A baseless claim has no anchor in
the evidence, whereas a conflicting claim engages a specific
passage only to assert the opposite, which motivates analyzing
the two separately. Surveys further organize hallucination by
whether it conflicts with the input, with world knowledge, or
with the model own earlier output, and a recurring theme is
that the phenomenon is best studied at a fine granularity, since
a response-level judgment hides which claim is at fault [2],
[3]. Reliable progress also depends on benchmarks. Early re-
sources annotated only whether a whole answer was faithful,
which restricts both training and evaluation to the response
level, while more recent resources annotate the offending spans
and their type, which is what enables the span-level detection
and per-type analysis we pursue [13]. Independent of the bench-
mark, trust in a detector depends on whether its decisions can
be inspected, so an explanation that points to the missing or
supporting evidence matters, especially where a person must
sign off on the output.
3.2 Language modeling and perplexity
An autoregressive language model factorizes the probability
of an answer y given a context c as a product of per-token
conditionals, p(y | c) = ∏t p(yt | y<t,c), where each conditional
is a softmax over the vocabulary. The perplexity of an answer
is the exponential of the mean surprisal,
PPL(y | c) = exp

−1
|y| ∑
t
log p(yt | y<t,c)

,
(1)
and it is the intrinsic baseline most often proposed for hallucina-
tion detection. However, perplexity underdetermines grounding.
An unsupported sentence that is well written and consistent with
the topic of the context is probable to the model and therefore
has low perplexity even though no passage supports it, and a
faithful sentence that paraphrases a passage unusually can have
high perplexity while being fully grounded. Perplexity mea-
sures probability under one context and cannot see whether that
probability came from the evidence or from the prior, which
is the tie our method breaks by comparing probabilities across
contexts. The same limitation applies to per-token variants of
the signal, such as the variance of surprisal across a sentence,
which captures how uneven the model confidence is but still
reads a single context. These intrinsic statistics can correlate
with hallucination in datasets where unsupported content hap-
pens to be less fluent, but the correlation is incidental and does
not survive when an unsupported sentence is written fluently,
which is precisely the case that matters in a deployed system
whose generator is competent. A grounding signal must there-
fore compare behavior across contexts, and the comparison
must be made at the granularity of the claim, since a fluent
answer can mix grounded and ungrounded sentences whose
average surprisal looks unremarkable.
3.3 RNIFS and invariant measures
An RNIFS is a finite family of maps {Fw}w∈V applied in a
stochastic sequence with selection probabilities {πw}. When
the maps are contractions and the selection is suitably regular,
Page 4 of 23

## Page 5

Research Article
Bouke, 2026
the induced sequence of distributions converges to a unique
invariant measure µ that satisfies the self-consistency relation
µ = ∑
w∈V
πw (Fw)∗µ,
(2)
where (Fw)∗µ is the pushforward of µ through Fw. Existence
and uniqueness follow from a fixed point argument on the space
of probability measures, in the spirit of the Banach fixed point
theorem applied to the Markov operator on the right of Eq. (2).
The relevant property for our purposes is stability, the way
the invariant measure responds to perturbation of the selection
probabilities. The RNIFS framework characterizes this stability
for nonlinear maps with stochastic selection, together with
the dimensional structure of the resulting attractors [16]. The
property we rely on is qualitative rather than a specific bound.
When a perturbation of the selection probabilities is localized,
in the sense that it changes the weights only where a removed
component was influential, the invariant measure changes only
on the corresponding region of the state space and is left close
to its original form elsewhere. This locality is what allows a
single global perturbation, the removal of one context unit, to
produce a localized effect that can be attributed to the parts
of the output that depended on that unit, and it is the formal
counterpart of the intuition that taking away a passage should
disturb the sentences built on it and leave the rest alone. The
same body of theory characterizes the geometric complexity of
these attractors, for example through their fractal dimension,
and shows that the dimension and the stability respond in a
controlled way to the parameters of the system [16]. We do
not use the dimensional results directly here, but they are the
reason we expect the response to a perturbation to be both
measurable and structured rather than chaotic, which is the
assumption underlying a detector that reads the size of that
response. We bring this machinery to bear on grounding by
treating the retrieved context as the component whose removal
perturbs the system and reading the induced movement of the
invariant measure as the grounding signal.
3.4 Divergence and evaluation
We compare two predictive distributions P and Q over the vo-
cabulary using the JSD,
JSD(P∥Q) = 1
2 KL(P∥M) + 1
2 KL(Q∥M),
M = 1
2(P+Q),
(3)
where KL is the Kullback-Leibler divergence. JSD is sym-
metric, bounded, and finite even when supports differ, which
makes it a stable measure of how far a predictive distribution
moves when the conditioning context changes. We prefer it
to the KL divergence alone, which is asymmetric and can di-
verge when one distribution assigns near-zero probability to
a token the other supports, a situation that arises routinely
when removing a passage sharply changes the plausible next
tokens. The boundedness of JSD also fixes the scale of the
divergence features, which we use later to argue that a single
operating threshold transfers across answers. For the decision
rule, we map the resulting features to a hallucination score with
a gradient-boosted decision tree classifier, implemented with
LightGBM [17], which fits an additive ensemble of shallow
trees by stage-wise optimization of a differentiable loss. Such
a classifier suits the small, heterogeneous, possibly nonlinear
tabular feature set we construct, needs no feature scaling, and
exposes feature importances, which keeps the detector light
and interpretable. We evaluate detection as binary classification
at the response and span levels and report the AUC, which is
threshold-free, together with the ROC curve. The AUC mea-
sures the probability that a randomly chosen hallucinated in-
stance receives a higher score than a randomly chosen grounded
one, so it summarizes the quality of the ranking without commit-
ting to a threshold and is insensitive to the class prior, which is
convenient when the evaluation set is balanced for measurement
but the deployment stream is not. The ROC curve complements
the single number by showing the full trade-off between the
true positive rate and the false positive rate, which is what a
practitioner consults when choosing an operating point for a
given review budget. Separately, span instances from the same
answer are dependent, so an evaluation that mixes them across
folds leaks information and inflates scores. A leakage-clean
protocol must group folds by answer, a discipline that connects
to the wider problem of data leakage inflating reported perfor-
mance in machine learning pipelines, where information from
the test set reaches training through preprocessing, resampling,
or grouping, and where the inflation is invisible unless the proto-
col is designed to prevent it. The effect is easy to underestimate,
since spans from one answer share lexical style, length, and
the same context, so a classifier that has seen some spans of an
answer can recognize the rest, and an ungrouped split would
report a score that reflects this recognition rather than genuine
detection. Table 2 summarizes the notation.
4 Problem Formulation
We are given a corpus of retrieval-augmented instances, each
a triple (q,c,y) of a query, a retrieved context, and a generated
answer, together with human span-level hallucination annota-
tions. The context is partitioned into units c = {s1,...,sK}, and
the answer is segmented into sentence spans y = (y1,...,ym)
with known character offsets. A labeling function assigns to
each span a binary label Y j ∈{0,1}, where Y j = 1 when the
span overlaps a human-annotated hallucination, and a type
τ j ∈{baseless,conflict,none} taken from the overlapping an-
notation. The response label is Y = maxjY j.
A detector is a scoring function f that maps an instance to a
real number, with higher scores indicating greater suspicion of
hallucination, instantiated as a response-level detector f(q,c,y)
and a span-level detector f(q,c,yj). Note the direction conven-
tion that we rely on throughout. The raw grounding-sensitivity
features of Section 5 increase with evidence dependence, so
they are larger for grounded spans. The final hallucination score
f is therefore defined after a monotone inversion, either learned
by the classifier or imposed by negating the standardized feature
sum, so that a higher f always means lower grounding. The
objective is to rank hallucinated instances above grounded ones,
measured by the AUC,
AUC(f) = Pr

f(u+) > f(u−)

,
(4)
for a positive instance u+ and a negative instance u−drawn at
random from the two classes. Because spans from one response
are dependent, estimating Eq. (4) at the span level must keep
Page 5 of 23

## Page 6

Research Article
Bouke, 2026
Table 2: Notation used throughout the paper.
Symbol
Meaning
q, c, y
query, retrieved context, generated answer
sk
the k-th context chunk, c = S
k sk
c\sk
context with chunk k removed
∅
empty (no) context
K
number of context chunks
yj, m
the j-th answer span and the number of
spans in y
|y|
number of tokens in span y
Y j, Y
span and response hallucination labels,
Y = maxjY j
τ j
hallucination type of span j (baseless, con-
flict, none)
f
hallucination score; a higher value is more
suspicious
ℓc,ℓ∅,ℓ(k)
answer log-likelihood under c, under no
context, and with chunk k removed
pct
predictive distribution at step t under c
jsd∅,jsdloo
no-context and leave-one-out divergence
features
ht, T
model hidden state at step t; decoding tem-
perature
A, V
answer length; vocabulary size
πw(·)
next-token selection probability
Fw, Tc
token update map; context-conditioned
transfer operator
µc
invariant measure under context c
H
space of hidden states
JSD
Jensen-Shannon divergence
W1, TV
Wasserstein-1 and total variation dis-
tances
¯L, D
average contraction modulus; one-step
spread of the maps
all spans of a response on the same side of every split, which
is the leakage-clean constraint we adopt. The task is to design
f so that it ranks well under this constraint and to make each
decision explainable, showing for a flagged span the absence
of a high-impact supporting unit and for a grounded span the
candidate supporting unit its likelihood most depends on.
5
Method
5.1 Decoding as a context-conditioned RNIFS
Let the model maintain a hidden state ht as it generates, and let
the retrieved context c shape the next-token distribution through
the logits zw(ht),
πw(ht;c) =
exp
 zw(ht;c)/T

∑w′ exp
 zw′(ht;c)/T
,
(5)
with temperature T.
Sampling a token w and updating
the state is a map ht+1 = Fw(ht) on the reachable state set
H ⊆Rd, so generation under c is a place-dependent RNIFS
{(Fw,πw(·;c))}w∈V whose selection probabilities in Eq. (5) de-
pend on c. Its distribution over states evolves under the Markov
transfer operator
(Tc ν)(·) = ∑
w∈V
Z
H πw(h;c)δFw(h)(·)dν(h),
(6)
whose fixed point is the invariant measure µc of Eq. (2), which
we call the attractor of grounded continuations. We make the
stability the method relies on explicit through the standard
average-contraction condition for such systems.
Assumption 1 (Average contraction and regular selection). Let
D = suph∈H maxw,w′∥Fw(h)−Fw′(h)∥be the one-step spread
of the maps. There are constants ℓand κ such that, uniformly
over h,h′ ∈H and over the contexts considered, the maps con-
tract on average, ∑w∈V πw(h;c)Lip(Fw) ≤ℓ, and the selection
distribution is Lipschitz in the state, TV
 π(h;c),π(h′;c)

≤
κ ∥h −h′∥, with combined modulus ¯L := ℓ+ Dκ < 1. Here
Lip(Fw) is the Lipschitz constant of Fw on H and TV is the
total variation distance.
This is the place-dependent form of contractivity, extended to
nonlinear maps with stochastic selection by the RNIFS analy-
sis [16]. The first condition makes the maps contract on average,
and the second keeps the selection probabilities from changing
too abruptly across nearby states, so that a place-dependent edit
cannot destabilize the operator. Together they make the transfer
operator a contraction, and under them the attractor exists, is
unique, and moves by no more than the perturbation a context
edit induces, which is exactly the property the detector reads.
Proposition 1 (Attractor movement is controlled by the mea-
sured divergence). Under Assumption 1, Tc is a contraction
of modulus ¯L on the probability measures over H under the
Wasserstein-1 distance W1, so a unique invariant measure µc
exists. Moreover, for any context edit c →c\s,
W1
 µc,µc\s

≤
D
1−¯L sup
h∈H
TV
 π(h;c),π(h;c\s)

≤
D
1−¯L sup
h∈H
q
2JSD
 π(h;c)∥π(h;c\s)

,
(7)
with D the one-step spread and TV the total variation as in
Assumption 1.
Proof sketch. Contractivity of Tc under W1 follows from the
place-dependent Hutchinson argument. For states h,h′, cou-
pling the images Tcδh and Tcδh′ through the shared token
index moves the matched mass by at most Lip(Fw)∥h −h′∥
per component and, where the two selection distributions
disagree, transports mass across atoms at most D apart,
so W1(Tcδh,Tcδh′) ≤ℓ∥h −h′∥+ DTV(π(h;c),π(h′;c)) ≤
(ℓ+ Dκ)∥h −h′∥= ¯L∥h −h′∥by Assumption 1.
Extend-
ing to general ν,ν′ through an optimal coupling gives
W1(Tcν,Tcν′) ≤¯LW1(ν,ν′), so a unique fixed point µc exists
by the Banach theorem. Both fixed points satisfy µ = T µ,
so W1(µc,µc\s) ≤W1(Tcµc,Tcµc\s) + W1(Tcµc\s,Tc\sµc\s) ≤
Page 6 of 23

## Page 7

Research Article
Bouke, 2026
Inputs: the question, the retrieved
sources, and the AI answer to check
Split the sources into pieces
Re-read with all sources, and
record the token likelihoods
and predictive distributions
Re-read with no sources, and
record the token likelihoods
and predictive distributions
Re-read with one piece removed at
a time, and record the token likeli-
hoods and predictive distributions
Measure likelihood drops and distribution shifts from
the all-sources reading when sources are removed
Combine the drops for each sentence
Turn the combined drops into a grounding-
sensitivity score for each sentence
Does the sentence show
high grounding sensitivity?
Supported
show the source it relies on
Unsupported
flag as a likely hallucination
yes
no
Figure 1: Overview of GASP. A fixed answer is re-scored under the full context, under no context, and under each retrieved chunk removed
in turn. The resulting per-sentence grounding-sensitivity score marks each sentence as supported, with its supporting source, or as a likely
hallucination.
¯LW1(µc,µc\s) + supν W1(Tcν,Tc\sν), which rearranges to the
factor 1/(1 −¯L).
The one-step gap supν W1(Tcν,Tc\sν) ≤
suphW1(Tcδh,Tc\sδh) compares two mixtures on the same
atoms {Fw(h)} with weights π(h;c) and π(h;c \ s), so it is
at most DTV(π(h;c),π(h;c \ s)). The last inequality is the
Pinsker bound applied to each half of the JSD, giving TV ≤
√
2JSD in nats.
Proposition 1 formalizes the intuition behind the signal under
an idealized setting. The movement of the attractor under a con-
text edit is bounded by the per-step JSD between the selection
distributions, the very quantity Eqs. (9) and (11) estimate at
the realized answer positions, amplified by the stability factor
1/(1−¯L). A grounded span sits where a specific edit makes this
divergence large, so its attractor moves and its likelihood falls,
whereas a free-running span sits where every edit leaves the
selection distribution, and hence the attractor, nearly fixed. The
likelihood features of Eqs. (8) and (10) read the same movement
through the realized tokens rather than the whole distribution.
Figure 8 in Section 8.6 shows this contrast on a real grounded
and a real hallucinated span, where the grounded span loses
most of its probability once the retrieved context is removed
while the hallucinated span is left almost unchanged. Figure 1
gives an overview of the procedure.
We do not claim Assumption 1 holds exactly for a transformer.
Real token-maps are neither globally contractive nor identical
across states, so Eq. (7) is a guide to the form of the signal rather
than a guarantee about a specific network, and the empirical
claims rest on the measurements. What the assumption buys is a
precise reason to measure a divergence between conditional gen-
eration measures rather than a surface statistic, and it is testable.
If decoding under a context behaves like a stable attractor, the
answer-token trajectory should occupy a low-dimensional set
rather than fill the state space, and the most influential con-
text edit should move that trajectory more than an irrelevant
one. Section 8.7 reports both, estimating the correlation dimen-
sion of the trajectory and the movement it undergoes when the
most influential chunk is removed against length, lexical, and
position controls.
Page 7 of 23

## Page 8

Research Article
Bouke, 2026
5.2 Grounding sensitivity
Definition 1 (Grounding sensitivity). For an answer span y
and a context unit s ⊆c, the grounding sensitivity of y with
respect to s is the distance between the conditional generation
measure under the full context, µc, and under the context with s
removed, µc\s, restricted to y. We instantiate this distance with
the reduction in the mean token log-likelihood of y and the JSD
between the per-token predictive distributions of y, evaluated
before and after removing s.
If y is grounded in s, removing s lowers the likelihood of y and
shifts its predictive distribution, so the grounding sensitivity is
large. If y depends on no unit, µc\s restricted to y is close to µc
for every s and the grounding sensitivity is small. The limiting
case is precise.
Proposition 2 (Insensitivity of free-running spans). If a span y
is generated independently of the context, so that p(yt | y<t,c) =
p(yt | y<t,c\s) for every token t of y and every unit s, then all
four grounding-sensitivity features of y defined in Section 5.3
are zero. Conversely, a nonzero value of any feature implies
that y depends on at least one context unit.
Proof sketch. Under the stated independence, each per-token
log-likelihood difference in Eqs. (8) and (10) is zero, and each
per-token JSD term in Eqs. (9) and (11) is the divergence of a
distribution from itself, which is zero. The aggregates, being
means and maxima of zeros, are zero. The converse is the
contrapositive.
Proposition 3 (Boundedness of the divergence features). With
the natural logarithm, every per-token JSD term satisfies 0 ≤
JSD(P∥Q) ≤ln2, so the aggregated divergence features lie in
[0,ln2], with ln2 ≈0.693.
Proof sketch. JSD is the mutual information between a fair
binary mixture indicator and a sample from the mixture, which
is at most one bit, equal to ln2 nats. A mean or maximum of
terms in [0,ln2] stays in [0,ln2].
Remark 1. In practice a hallucinated span is not perfectly
independent of the context, because topic and style still leak
through, so its grounding sensitivity is small but not exactly
zero. The converse in Proposition 2 is also weaker than it may
first appear. A nonzero feature shows only that the span reacts
to removing some unit, and removing a chunk perturbs more
than grounding alone, since it shortens the context, shifts the
positions of the later tokens, alters formatting and any list or
table structure, can truncate the context differently, and can
break discourse coherence, so part of the measured sensitivity
may reflect these confounds rather than genuine support. The
features are therefore indicative rather than decisive, which
is why we learn a threshold on the continuous signal instead
of testing exact independence, and why we report empirical
ranking performance rather than treating the proposition as a
detector on its own. The divergence features share a common
scale by Proposition 3, and the likelihood features are differ-
ences that remove the dependence on absolute fluency, so the
features are reasonably comparable across answers.
5.3 Features
Let log p(y | c) = ∑t log p(yt | y<t,c) be the answer log-
likelihood under context c, and let pc
t be the predictive dis-
tribution at position t under context c. We re-score the fixed
answer under the full context c, the empty context ∅, and the
context with the k-th chunk removed c \ sk for k = 1,...,K.
From these we form four features per span y. The reliance
on context as a whole is the gap between the full-context and
no-context log-likelihoods,
gap(y) =
1
|y| ∑
t∈y
h
log p(yt | y<t,c)−log p(yt | y<t,∅)
i
, (8)
with distributional counterpart the mean JSD between full-
context and no-context predictions,
jsd∅(y) =
1
|y| ∑
t∈y
JSD
 pc
t ∥p∅
t

.
(9)
The single most supportive context unit is captured by the max-
imum leave-one-out likelihood drop,
drop(y) = max
1≤k≤K
1
|y| ∑
t∈y
h
log p(yt | y<t,c)
−log p(yt | y<t,c\sk)
i
,
(10)
with distributional counterpart
jsdloo(y) = max
1≤k≤K
1
|y| ∑
t∈y
JSD
 pc
t ∥pc\sk
t

.
(11)
Equations (8) and (9) measure dependence on the whole context,
while Eqs. (10) and (11) localize dependence to the single most
important unit, which makes the method robust when only one
passage supports a span. The likelihood-based and divergence-
based features are complementary by design. The likelihood
drop asks how much less probable the realized answer tokens
become when a unit is removed, which is the quantity most
directly tied to whether the model used that unit, while the di-
vergence asks how far the whole predictive distribution moves,
which remains informative even when the realized answer al-
ready had low probability under the full context. Reporting both
lets the classifier weight the realized-token view and the whole-
distribution view according to the data rather than committing
to one in advance.
5.4 Aggregation to spans and responses
All four conditions share the same answer tokenization, so
the per-token quantities are computed once per condition and
then aggregated. Spans are sentences obtained by splitting the
answer at sentence-final punctuation followed by whitespace,
with the character offset of each sentence retained. Each token
is assigned to the sentence that contains its starting offset, and
the per-token values are averaged over a sentence for span-
level features and over the whole answer for response-level
features. The same offsets align spans with the benchmark
character-level annotations, so a sentence inherits the label and
type of any annotated span it overlaps. This reuse means span-
level and response-level detection share a single set of forward
passes, which is the property that makes per-sentence scoring
Page 8 of 23

## Page 9

Research Article
Bouke, 2026
affordable, since the finer granularity adds only arithmetic over
the already-computed per-token quantities and no further model
calls. The aggregation is a simple average over the tokens of
a span, which keeps the span features on the same scale as the
response features and avoids giving long sentences an advantage
that a sum would introduce.
5.5 Detection and explanation
The four grounding-sensitivity features of Section 5.3 are larger
for grounded spans, so the hallucination score is a decreasing
function of them, and the alarm is low sensitivity, not high. The
default detector, GASP-threshold, is training-free. It standard-
izes the four features and thresholds their negated sum, so it
needs no labeled data, and we report it as the default detec-
tor throughout. An optional supervised variant, GASP-trained,
replaces the fixed sum with a gradient-boosted decision tree
classifier [17] that maps the feature vector of a span or a re-
sponse to a hallucination score and learns the direction from
labels, and a combined variant, GASP+base, adds perplexity
and length to that classifier. Both are reported for compari-
son with the default. We refer to the complete detector, the
grounding-sensitivity features together with either the threshold
or the classifier, as GASP.
The explanation is a byproduct of the same computation. Be-
cause the score is built from the effect of removing context
units, the unit attaining the maximum in Eq. (10) is returned as
the supporting evidence for a grounded span, and the absence
of any unit with a large drop is the reason a span is flagged.
The explanation is therefore mechanism-consistent, since the
supporting unit is the very perturbation that most changed the
model behavior on the span, so the reason shown to a user is
computed from the same signal that produced the score rather
than a post hoc rationalization from a separate method, in con-
trast to detectors that emit a score and then attach an explanation
that can diverge from the mechanism that produced it. For a
flagged span the explanation is the absence of any unit with
a large effect, which a reviewer can confirm by reading the
context, and for a grounded span it is the single passage whose
removal most reduced the span likelihood, which serves as a
candidate supporting unit rather than a verified citation, since
we do not have gold evidence alignment and evaluate the at-
tribution only against automatic controls in Section 8.5. The
supervised variant also exposes its feature importances, so a
practitioner can see which of the four sensitivity features drove
a decision in aggregate, separately from the per-span supporting
passage.
Figure 2 illustrates the signal. A grounded sentence draws on
a specific passage, so removing that passage lowers the sen-
tence’s token log-likelihood and shifts its predictive distribution,
producing a large likelihood drop and divergence for the chunk
that carried the support. A hallucinated sentence does not draw
on any passage, so removing any chunk leaves its likelihood
and predictive distribution almost unchanged, producing small
values across all removals. The detector therefore assigns high
grounding sensitivity to the grounded sentence and flags the
hallucinated one, and it returns the supporting passage as the
explanation for the grounded sentence. This behavior is the
qualitative counterpart of Proposition 2, in which a grounded
Grounded
sentence
retrieved context
remove support
likelihood
with
without
large drop
high
sensitivity
Hallucinated
sentence
retrieved context
remove any chunk
likelihood
with
without
small drop
low
sensitivity
Figure 2: Schematic of the grounding-sensitivity signal. Removing
the supporting chunk sharply lowers the likelihood of a grounded sen-
tence, a large drop that yields high sensitivity, but barely changes a
hallucinated sentence, a small drop that yields low sensitivity.
sentence has nonzero sensitivity to a specific unit while a free-
running sentence is insensitive to all units.
5.6 Algorithm and complexity
Algorithm 1 summarizes the procedure, and Table 3 lists the
cost of each condition. For a response with answer length
A, vocabulary V, and K chunks, the method performs K + 2
scoring passes over the fixed answer, with divergence terms
costing O(AV) per condition, and no generation or training for
feature construction. Several choices warrant justification. We
chunk the context rather than removing single sentences, which
bounds the number of passes at K + 2 at the price of coarser
localization. The scoring model need not be the generator, since
the method requires only a probabilistic model of the answer
under a context, so a small separate scorer can audit answers
from a larger or hosted generator provided it follows the context
well enough to react to its removal.
The maximum over chunks is one of several reasonable ag-
gregations, and the choice reflects an assumption about how
support is distributed. A maximum suits single-passage support
and is robust to the many irrelevant removals that leave a span
unchanged, a mean would measure diffuse reliance but would
be dominated by the irrelevant removals, and a top-few sum sits
between the two when support is split across a small number
of passages. The no-context condition is the limiting case of
removing all units at once, so it is the natural partner of the
single-unit maximum, and the two together bracket the range
from concentrated to diffuse support. We keep the feature set
deliberately small, four sensitivity features plus the baselines,
both to limit the risk of overfitting on a few hundred responses
and to keep the detector interpretable, since each feature has a
direct reading in terms of how the span responds to removing the
whole context or the single most important chunk. A further re-
finement retains the sign of the likelihood change rather than its
magnitude, since a passage that a span contradicts and a passage
that a span relies on both change the likelihood when removed,
but in directions that may differ. The AUC is threshold-free, but
a deployment must choose an operating point, and because the
divergence part of the signal is bounded and comparable across
spans by Proposition 3, a single threshold transfers reasonably,
with a low threshold favoring recall for high-stakes review and
a high threshold favoring precision for automated gating.
6 Experimental Setup
Page 9 of 23

## Page 10

Research Article
Bouke, 2026
Algorithm 1: Grounding-sensitivity scoring of a RAG
answer.
Input: answer y, query q, context c split into chunks
s1,...,sK, model M
Output: per-span grounding-sensitivity features and
supporting unit
1 ℓc,Pc ←score y with M under (q,c);
2 ℓ∅,P∅←score y with M under (q,∅);
3 for k ←1 to K do
4
ℓ(k),P(k) ←score y with M under (q,c\sk);
5 end
6 compute per-token gap, jsd∅, leave-one-out drop,
jsdloo from Eqs. (8)–(11);
7 foreach sentence span y′ in y do
8
aggregate per-token features over the tokens of y′;
9
k∗←argmaxk likelihood drop on y′;
10
emit features of y′ and supporting unit sk∗;
11 end
Table 3: Computational cost per response, with answer length A,
vocabulary V, and K context chunks.
Condition
Forward passes
Divergence cost
Full context
1
n/a
No context
1
O(AV)
Leave-one-out
K
O(KAV)
Total
K +2
O(KAV)
6.1 Datasets
We evaluate on RAGTruth [13], a benchmark of retrieval-
augmented answers with human span-level hallucination anno-
tations and a type label for each annotated span. We use the two
task types whose answers are long enough for stable estimation,
summarization and data-to-text, and we sample a class-balanced
set of 200 hallucinated and 200 clean responses. The same 400
responses, with identical text and gold span labels, are scored
by all three models. Only the scoring model changes, so the
comparison across scorers is on one fixed evaluation set. The
generators that produced these answers in RAGTruth (for ex-
ample GPT-4) are distinct from our scorers, which is why we
describe the scorer as a probabilistic model of the answer rather
than the generator. Sentence segmentation is text-based, split-
ting the answer at sentence-final punctuation, so it is identical
across scorers. The small difference in sentence counts, 2,586
for the Qwen tokenizer and 2,550 for SmolLM2, comes only
from the 200-token answer cap landing at a slightly different
text position under each tokenizer.
We use three benchmarks to answer three different questions,
so their roles and their depth of treatment differ by design rather
than by convenience. RAGTruth is the primary benchmark
and carries the full study, because it is the only one of the
three with human span-level hallucination annotations together
with a hallucination-type label, which is what supports the
per-type, per-task, baseline, ablation, attribution, and attractor
analyses. On it we ask not only whether the signal detects
hallucination but how it behaves. TofuEval then asks whether
Table 4: Composition of the three evaluation benchmarks. RAGTruth
covers summarization and data-to-text, TofuEval meeting summariza-
tion, and RAGBench short-answer question answering. RAGTruth and
RAGBench are class-balanced by sampling, while TofuEval takes the
natural distribution over its summaries. Sentence counts use the Qwen
tokenizer.
Quantity
RAGTruth
TofuEval
RAGBench
Responses
400
884
797
hallucinated
200
309
400
clean
200
575
397
Sentences
2,586
2,401
3,858
hallucinated
313
440
778
grounded
2,273
1,961
3,080
Scorers
3
3
2
the signal reaches beyond RAGTruth to a different domain at the
same span-level granularity, which requires a second annotated
benchmark but not a repeat of every analysis. RAGBench asks
whether it reaches a different task type, short-answer question
answering, where the answer need not be constructed from the
retrieved context. The three are therefore not interchangeable.
RAGTruth establishes the method, TofuEval tests its reach
across domain, and RAGBench probes its reach across task
type, and each is used only to the depth its question requires.
TofuEval, in its MeetingBank portion [18], [19], provides
sentence-level factual-consistency labels for topic-focused sum-
maries of meeting transcripts, a different domain and anno-
tation scheme from RAGTruth. We map each summary into
the same format, the transcript as context, the topic as query,
and each summary sentence as an answer span with a binary
label, raise the context cap to 1500 tokens so every transcript
fits, and score all three models over 884 summaries and 2,401
sentences. RAGBench [20] is a multi-domain benchmark of
retrieval-augmented question answering. We pool the test splits
of six domains spanning biomedical, financial, technical, and
open-domain QA into a class-balanced set of 797 responses
and 3,858 sentences with genuine per-sentence unsupported-
sentence labels. Because its role is to probe the reach of the
method rather than to re-establish it, the two Qwen scorers
are sufficient and we report a single span-level table. Table 4
summarizes the composition of the three benchmarks, and Sec-
tions 7.6 and 7.7 report the TofuEval and RAGBench results.
6.2 Models
We score answers with three instruction-tuned models. Two,
Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct [21], come
from one family and differ in scale, which tests robustness to
model size, and the third, SmolLM2-1.7B-Instruct [22], comes
from a different family, which tests robustness across lineages.
All three run on a single consumer graphics processing unit
(GPU) with 6 gigabytes of memory, and scoring uses forward
passes only and no generation.
6.3 Baselines
We compare grounding sensitivity against perplexity, the
mean answer surprisal under the full context, and length,
the number of answer tokens.
For entailment we re-
Page 10 of 23

## Page 11

Research Article
Bouke, 2026
port the
cross-encoder/nli-deberta-v3-small
and cross-encoder/nli-deberta-v3-large mod-
els,1 scoring the probability that a premise entails a sentence
hypothesis, so that the comparison is not against a deliberately
weak verifier. Because a long context truncated to the cross-
encoder limit of 512 tokens can cut the supporting evidence,
we report the entailment two ways, once with the whole con-
text as premise and once as the maximum entailment over the
same K = 5 chunks used by the detector, which lets the verifier
attend to the single most relevant passage. We further compare
against a SelfCheckGPT-style self-consistency baseline [7] that
draws N = 4 stochastic regenerations of the answer from the
same scorer under the same context, with temperature 1.0 and
nucleus sampling at p = 0.95, and scores each answer sentence
by its average contradiction probability against the four samples
under the compact NLI model, which is the sentence-level NLI
variant of SelfCheckGPT. We also report a combined feature
set that adds perplexity and length to the grounding-sensitivity
features.
The baselines above represent the detection families that can be
run under one unified leakage-clean protocol, namely intrinsic
uncertainty through perplexity, entailment verification through
the NLI cross-encoders, and sampling agreement through self-
consistency. Two closely related recent lines, the mechanistic
detectors and the supervised RAG judges, are cited but not eval-
uated head to head, because they operate in a different regime
from the black-box, training-free setting studied here. The
mechanistic detectors that separate context use from parametric
knowledge, such as ReDeEP and LUMINA [11], [12], read the
generator’s internal states and therefore assume white-box ac-
cess to the model that produced the answer, whereas grounding
sensitivity is computed from output likelihoods alone, so a faith-
ful re-implementation would require their specific generators
and activations rather than the scoring interface used here. The
supervised detector LettuceDetect [15] trains a dedicated judg-
ing model on RAGTruth, so its released weights have already
seen the corpus we evaluate on, and a fair inclusion under the
grouped folds would require retraining it per fold rather than
reusing a model fit on data that overlaps the test partitions. It
also represents the trained-judge paradigm that the training-free
variant is meant to offer an alternative to. We therefore position
the present work against these methods conceptually in Sec-
tion 2, and we do not place their reported numbers in the same
tables, because those numbers come from different generators,
splits, and metrics and are not directly comparable under our
protocol.
6.4 Protocol and metrics
We report AUC under five-fold cross-validation with a gradient-
boosted decision tree classifier. For the response level we use
stratified folds, and for the span level we use stratified folds
grouped by response, so that sentences from one answer never
appear in both the training and the test partitions. For the per-
type analysis we evaluate clean sentences against sentences
of a single type at a time under the same grouped protocol.
1Hugging Face:
https://huggingface.co/cross-encoder/
nli-deberta-v3-small
and
https://huggingface.co/
cross-encoder/nli-deberta-v3-large.
Confidence intervals are 95% percentile bootstrap intervals
over 1000 resamples, drawn by response at the span level so the
grouping is preserved, and significance against perplexity uses
a paired bootstrap on the same resamples. The point estimate in
every table is the mean AUC over these resamples rather than
the single value computed once on the out-of-fold predictions,
a standard bootstrap summary that can differ from the latter by
around a thousandth. Both the five-fold split and the number of
bootstrap resamples are conventional choices for this kind of
evaluation.
6.5 Implementation
Context is truncated to 700 tokens and the answer to 200 tokens,
the context is split into K = 5 chunks by sentence grouping for
the leave-one-out perturbations, and per-token divergences are
computed over the full vocabulary. Scoring an answer under a
condition concatenates the formatted prompt for that condition
with the answer tokens, runs one forward pass, and reads the per-
token log-probabilities and predictive distributions at the answer
positions. The classifier uses fixed conventional settings, 300
trees with a learning rate of 0.05 and 31 leaves, which were not
tuned on the evaluation data, since the recommended detector is
the training-free threshold and the classifier is reported only for
comparison. All experiments use fixed random seeds. The same
answer tokenization is reused across all conditions, so the per-
token quantities align position by position and can be compared
directly, and no fine-tuning or gradient computation is involved.
Sentence boundaries and the token-to-sentence assignment use
the character offsets from the tokenizer, so the span aggregation
is exact rather than approximate, and fragments shorter than
a few tokens are skipped because their feature estimates are
unstable.
Table 5 lists the full configuration used for every reported result,
and Table 6 restates the four grounding-sensitivity features and
the three baselines.
6.6 Reproducibility
The only model calls are the K+2 scoring passes, and every-
thing after is arithmetic over the stored per-token quantities,
so the per-answer latency is the time of K+2 forward passes
of a small model over a few hundred tokens, modest relative
to generating the answer, and passes for different answers are
independent, so throughput scales with parallelism. The diver-
gence terms are computed over the full vocabulary, which is
the most memory-intensive part. When memory is constrained,
they can be computed in vocabulary blocks or restricted to the
top-probability tokens with little loss, which does not affect the
likelihood-based features. Because the answer is never regen-
erated, the procedure is deterministic given the model and the
inputs and does not depend on a sampling temperature. The
sampling of responses is deterministic given the seed and every
step is seeded, so a re-run reproduces the numbers exactly on
the same hardware, with differences on other hardware limited
to the last digits from floating-point summation order. To re-
produce a reported table, one runs the scoring over the sampled
responses to produce the per-token features, aggregates them
to the span and response levels, and runs the grouped cross-
validation with the listed classifier settings, and the figures are
produced from the same feature files by the plotting scripts.
Page 11 of 23

## Page 12

Research Article
Bouke, 2026
Table 5: Configuration for all experiments. The pipeline runs in two
stages, GPU scoring with PyTorch, transformers, and Hugging Face
datasets, then classifier training and metric computation with scikit-
learn and LightGBM.
Component
Setting
Scorer models
Qwen2.5-0.5B / 1.5B-Instruct,
SmolLM2-1.7B-Instruct
Entailment models
deberta-v3-small / large NLI
Self-consistency samples
N = 4
Max context tokens
700
Max answer tokens
200
Context chunks K
5 (sensitivity: 3, 5, 10)
Classifier
gradient-boosted trees
trees
300
learning rate
0.05
leaves
31
Cross-validation
5-fold
span level
grouped by response
Bootstrap resamples
1000, grouped
Random seed
fixed
GPU
NVIDIA RTX 3060 Laptop, 6 GB
CPU
Intel Core i7-11800H
System RAM
32 GB
Operating system
Windows 11
Python
3.14
PyTorch
2.6.0 (CUDA 12.4)
transformers
5.3.0
Hugging Face datasets
4.7.0
scikit-learn
1.8.0
LightGBM
4.6.0
Table 6: Summary of features and baselines.
Name
Definition
gap
full vs no-context log-likelihood, Eq. (8)
jsd∅
full vs no-context JSD, Eq. (9)
drop
max leave-one-out drop, Eq. (10)
jsdloo
max leave-one-out JSD, Eq. (11)
perplexity
Eq. (1) under full context
length
number of answer tokens
NLI entailment
entailment probability, context vs answer
7
Results
7.1 Response-level detection
Table 7 reports response-level AUC with 95% bootstrap con-
fidence intervals for three scorers spanning two model fam-
ilies.
The training-free default, GASP-threshold, reaches
0.745, 0.713, and 0.741 for Qwen2.5-0.5B, Qwen2.5-1.5B, and
SmolLM2-1.7B, and the trained classifier reaches 0.706, 0.726,
Figure 3: Span-level AUC by feature set for the three scorers, with
bootstrap 95% confidence intervals as error bars. GASP-thr is the
training-free threshold-only default, GASP-trn the trained classifier, and
GASP+base the trained combined set. All sit well above perplexity,
length, and the chance line (dashed).
Figure 4: Response-level AUC by feature set for the three scorers,
with bootstrap 95% confidence intervals as error bars. GASP-thr is
the training-free threshold-only default, GASP-trn the trained classifier,
GASP+base the trained combined set. Dashed line: chance.
and 0.716. Both are well above perplexity, which attains 0.581,
0.624, and 0.578, and far above the length baseline near 0.55.
The trained classifier’s gap over perplexity is +0.10 to +0.14
and is significant under a paired bootstrap on all three scorers,
with p < 0.001, p = 0.002, and p = 0.001. Adding the base-
lines to the trained classifier changes the score only marginally,
so grounding sensitivity already carries the signal at the re-
sponse level. Figure 4 visualizes the comparison.
7.2 Span-level detection
Table 8 reports span-level AUC under the leakage-clean,
response-grouped protocol, again with bootstrap confidence in-
tervals. The default GASP-threshold reaches 0.672, 0.673, and
0.681 across the three scorers and the trained classifier reaches
0.635, 0.645, and 0.657, above perplexity (0.536, 0.565, 0.615)
and the length baseline near 0.55. The grounding gap over per-
plexity is significant on all three scorers under a paired bootstrap
(p < 0.001, p < 0.001, and p = 0.022). The combined trained
set improves the span-level score to between 0.676 and 0.702,
and a well-configured chunk-level NLI verifier reaches 0.677
(Section 8.2), so grounding sensitivity and a strong entailment
verifier are close at this granularity. Figure 3 shows the same
comparison, and Figure 5 shows the span-level ROC curves
for the three scorers, where the grounding curve stays above
perplexity and whole-context NLI entailment across most of
the operating range.
Page 12 of 23

## Page 13

Research Article
Bouke, 2026
Table 7: Response-level hallucination detection AUC with 95% bootstrap confidence intervals over 1000 resamples, three scorers. GASP-
threshold (default) is the training-free variant. Best mean per column in bold.
Feature set
Qwen2.5-0.5B
Qwen2.5-1.5B
SmolLM2-1.7B
Perplexity
0.581 [0.524, 0.638]
0.624 [0.572, 0.675]
0.578 [0.522, 0.630]
Length
0.553 [0.496, 0.611]
0.553 [0.497, 0.609]
0.539 [0.483, 0.593]
GASP-threshold (default)
0.745 [0.692, 0.793]
0.713 [0.661, 0.765]
0.741 [0.691, 0.788]
GASP-trained
0.706 [0.654, 0.757]
0.726 [0.676, 0.776]
0.716 [0.667, 0.766]
GASP+base (trained)
0.731 [0.679, 0.779]
0.713 [0.658, 0.760]
0.717 [0.668, 0.765]
Table 8: Span-level (sentence) AUC under the leakage-clean protocol with folds grouped by response, with 95% bootstrap confidence intervals.
GASP-threshold (default) is the training-free variant. The lower block separates the two hallucination types (single type versus clean). Best mean
per column in bold.
Feature set
Qwen2.5-0.5B
Qwen2.5-1.5B
SmolLM2-1.7B
Perplexity
0.536 [0.506, 0.568]
0.565 [0.532, 0.599]
0.615 [0.584, 0.646]
Length
0.547 [0.510, 0.583]
0.548 [0.512, 0.582]
0.523 [0.489, 0.558]
GASP-threshold (default)
0.672 [0.641, 0.701]
0.673 [0.645, 0.700]
0.681 [0.655, 0.708]
GASP-trained
0.635 [0.608, 0.664]
0.645 [0.615, 0.675]
0.657 [0.624, 0.687]
GASP+base (trained)
0.676 [0.647, 0.703]
0.678 [0.648, 0.707]
0.702 [0.673, 0.731]
baseless vs clean
0.620
0.638
0.641
conflict vs clean
0.685
0.704
0.681
7.3 Robustness across scorers
The grounding-sensitivity AUC is stable across the three tested
scorers at both granularities, and the three fall within each
other’s confidence intervals, which suggests the signal reflects
the grounding relationship rather than a particular model. The
three span two families, since SmolLM2-1.7B is from a dif-
ferent lineage than the two Qwen models, so the result is not
purely within-family, though three scorers remain a small sam-
ple and families such as Llama, Mistral, and Gemma, and larger
scorers, would be needed to claim broad invariance. The mar-
gin over perplexity is not equally large on every scorer. On
SmolLM2 the perplexity baseline is stronger at the span level,
0.615, and the grounding gap is correspondingly smaller, 0.657
against 0.615 and still significant at p = 0.022, whereas on the
Qwen scorers perplexity is weaker and the gap is wider, so the
advantage is consistent in sign but not identical in size across
scorers. Among the other baselines, the length baseline is far
below grounding sensitivity for every scorer, so the detector is
not tracking span length, and the whole-context entailment base-
line stays below grounding sensitivity, though a chunk-level
entailment verifier is competitive (Section 8.2). This robustness
has a specific scope. It indicates that across the scorers tested,
which vary in size and family, the grounding signal is essen-
tially unchanged, but not that an arbitrarily weak scorer would
suffice, because a scorer that fails to attend to the context would
react little to its removal even for grounded spans, which would
compress the signal toward chance. The three scorers are all
competent at following the context, so the property the method
needs is competence at using the context rather than raw scale,
and that property is already present in small instruction-tuned
models.
7.4 Per-type analysis
The lower block of Table 8 separates the two hallucination types.
Grounding sensitivity detects evident-conflict sentences slightly
better than baseless-information sentences on all three scor-
ers, with conflict at 0.685, 0.704, and 0.681 against baseless at
0.620, 0.638, and 0.641. Both types are detected well above
chance, the ordering is consistent across scorers, and the gap
is modest, so grounding sensitivity detects both unsupported
additions and contradictions. Detecting both matters, since
a detector that only caught baseless additions would miss the
contradictions that are often the more dangerous errors in a high-
stakes setting, and a detector that only caught contradictions
would miss the unsupported additions that are more common.
The consistency of the ordering across the three scorers also
makes it less likely that the per-type difference is a sampling ar-
tifact, and it motivates the direction-aware refinement discussed
later.
7.5 Detection by task type
RAGTruth mixes two task types, and a single-benchmark result
is stronger if the signal is not tuned to one of them. Table 9
splits the trained grounding detector by task under the same
leakage-clean protocol. The signal is present on both, with data-
to-text detected more consistently, response AUC 0.62 to 0.67
and span AUC 0.58 to 0.66, than summarization, whose values
are more variable and, for the 1.5B scorer at the span level,
fall to 0.535 with a confidence interval that includes chance.
Summarization also contributes far fewer hallucinated spans
to the sample, which widens its intervals. The split shows that
Page 13 of 23

## Page 14

Research Article
Bouke, 2026
Table 9: Grounding detection AUC by RAGTruth task type, leakage-
clean protocol, using the trained detector (GASP-trained) as a consis-
tent reference across tasks. These are point estimates with wide confi-
dence intervals, since each task contributes roughly 200 responses.
Level
Task
Qwen-0.5B
Qwen-1.5B
SmolLM2
Response
Summary
0.554
0.669
0.636
Response
Data2txt
0.661
0.619
0.669
Span
Summary
0.563
0.535
0.646
Span
Data2txt
0.577
0.629
0.660
the method generalizes across task type but is not uniformly
strong, and that data-to-text, where an answer restates structured
fields, exposes the dependence on evidence more cleanly than
free-form summarization.
7.6 TofuEval
Table 10 reports the TofuEval result over 884 summaries and
2,401 sentences, scored with the identical pipeline and features
under the leakage-clean protocol. The benchmark and its map-
ping to our format are described in Section 6.1. At the span level
GASP-threshold reaches 0.693 on all three scorers, computed
independently per scorer and coinciding only after rounding
to three decimals, close to and slightly above its RAGTruth
values, and the trained and combined variants sit between 0.63
and 0.68, all significantly above perplexity, which stays at 0.52
to 0.56, with the grounding gap over perplexity significant on
every scorer (p < 0.001, except p = 0.005 at the response level
for the 1.5B scorer). Length is near chance. The training-free
threshold-only default again matches or exceeds the trained clas-
sifier. Response-level detection is a little lower, 0.61 to 0.66,
and remains significantly above perplexity. Taken together, two
benchmarks in two domains, with consistent behavior and the
same recommended default detector, is stronger evidence of
external validity than either alone, though two datasets remain
a limited sample.
7.7 RAGBench
RAGBench probes the complementary question of where the
signal does not hold, on multi-domain short-answer QA. The
benchmark and its pooling across six domains are described
in Section 6.1. At the span level (Table 11), unlike the sum-
marization benchmarks, grounding sensitivity is comparable to
perplexity rather than better, with the trained detector at 0.60
and the threshold-only variant at 0.57 against perplexity at 0.61
to 0.62, and the grounding gap over perplexity is not significant
(p > 0.5). The combined set reaches 0.67 to 0.68, above either
alone, so the grounding features still add complementary infor-
mation, but on this benchmark perplexity carries most of the
signal. On the labeling side, RAGBench annotates each exam-
ple with a set of unsupported response-sentence keys, so the
span-level labels are genuine sentence-level labels obtained the
same way as for RAGTruth and TofuEval, and the span-level
AUC is a like-for-like comparison across the three benchmarks.
We report the span level rather than the response level because
the response-level adherence label is coarse, marking a whole
multi-sentence response unsupported if any single sentence is,
which is poorly matched to features averaged over an otherwise
grounded response.
This is a scope boundary rather than a failure of the method. The
direction of the signal is unchanged, since grounded sentences
remain more sensitive to context removal than hallucinated ones,
but its magnitude is small on short-answer, knowledge-intensive
QA. A plausible reason is that when a model can answer a fac-
tual question from its parametric knowledge, a grounded answer
is not strongly sensitive to removing the retrieved passage, be-
cause the model would produce it anyway, and perplexity is
already discriminative in that regime. Grounding sensitivity
is therefore best suited to settings where the answer must be
constructed from the provided context, such as summarization
and data-to-text, where RAGTruth and TofuEval show the clear
gains, and it adds little over perplexity on short-answer retrieval
QA.
8 Analysis and Diagnostics
Having established the detection results, this section examines
the detector in more detail.
8.1 Feature analysis
Figure 6 shows that the grounding sensitivity of grounded sen-
tences is shifted toward larger context-removal effects than that
of hallucinated sentences, with overlap in the middle that is
consistent with the moderate absolute AUC. Figure 7 shows
that no single feature dominates the combined classifier on any
scorer, since the four grounding features carry comparable gain
and length is the weakest throughout. The relative roles of
the two views are clearer in the ablation of Section 8.3, where
the no-context features are the strongest single group and the
leave-one-out features add a smaller complementary signal at
the span level. This is consistent with the design intuition that
a grounded sentence usually leans on one passage, which the
leave-one-out contrast localizes, while total reliance on the con-
text is captured by the no-context contrast. The combined model
being strongest at the span level, where the single-passage view
and the whole-context view are both available, indicates that
the two signals capture different aspects of dependence rather
than redundant ones, and it explains why neither alone reaches
the combined performance.
8.2 Comparison with baselines
Table 12 compares grounding sensitivity at the span level,
on the Qwen2.5-1.5B scorer, against the entailment and self-
consistency baselines. The large NLI cross-encoder improves
on the small one, from 0.532 to 0.605 with the whole con-
text, and to 0.677 when it is allowed to attend to the single
best of the K = 5 chunks rather than a premise truncated at
512 tokens. Configured this way, the entailment verifier es-
sentially matches GASP at the span level, 0.677 against the
threshold-only 0.673 and the trained 0.645, so GASP does not
exceed a well-configured chunk-level NLI verifier at this granu-
larity, and the two lie within each other’s confidence intervals.
GASP instead requires no separate verifier model, uses the
same scorer that computes the likelihood, extends to the re-
sponse level, where the entailment baselines are undefined, and
produces the attribution evaluated in Section 8.5. By contrast,
the self-consistency baseline reaches 0.486, near chance in this
Page 14 of 23

## Page 15

Research Article
Bouke, 2026
Table 10: TofuEval-MeetingBank, sentence-level factual consistency for topic-focused meeting summaries, under the leakage-clean protocol
with 95% bootstrap confidence intervals. GASP-threshold is the training-free default. Best mean per column in bold.
Feature set
Qwen-0.5B
Qwen-1.5B
SmolLM2
Response level
Perplexity
0.521 [0.481, 0.562]
0.548 [0.509, 0.586]
0.516 [0.479, 0.552]
GASP-threshold (default)
0.625 [0.584, 0.663]
0.623 [0.581, 0.663]
0.614 [0.575, 0.653]
GASP-trained
0.632 [0.594, 0.669]
0.627 [0.590, 0.665]
0.641 [0.605, 0.680]
GASP+base (trained)
0.659 [0.620, 0.697]
0.633 [0.594, 0.670]
0.656 [0.614, 0.695]
Span level
Perplexity
0.555 [0.525, 0.587]
0.516 [0.487, 0.545]
0.524 [0.497, 0.553]
Length
0.515 [0.485, 0.544]
0.515 [0.484, 0.545]
0.535 [0.503, 0.565]
GASP-threshold (default)
0.693 [0.659, 0.726]
0.693 [0.661, 0.726]
0.693 [0.658, 0.726]
GASP-trained
0.656 [0.620, 0.691]
0.642 [0.607, 0.676]
0.631 [0.593, 0.664]
GASP+base (trained)
0.672 [0.636, 0.705]
0.675 [0.642, 0.709]
0.680 [0.646, 0.714]
Table 11: Scope probe on RAGBench, a multi-domain short-answer QA benchmark (797 responses, 3,858 sentences, two Qwen scorers).
Span-level AUC with 95% bootstrap confidence intervals. Unlike the summarization benchmarks, grounding sensitivity is comparable to, not
better than, perplexity here (grounding versus perplexity p > 0.5).
Method
Qwen-0.5B
Qwen-1.5B
Perplexity
0.613 [0.590, 0.634]
0.619 [0.599, 0.642]
GASP-threshold
0.576 [0.540, 0.610]
0.573 [0.540, 0.607]
GASP-trained
0.595 [0.572, 0.618]
0.596 [0.573, 0.618]
GASP+base (trained)
0.668 [0.646, 0.690]
0.679 [0.659, 0.700]
Figure 5: Span-level ROC for the three scorers, comparing grounding sensitivity with perplexity and whole-context NLI entailment. The
whole-context NLI baseline does not depend on the scorer, so it is drawn as a shared reference in each panel. A chunk-level entailment verifier,
reported in Table 12, is competitive with grounding sensitivity and is not shown here.
grounded-summarization setting, at a cost of N full generations
per answer rather than K+2 scoring passes over a fixed answer.
The two GASP variants are also compared. The training-free
threshold-only variant, a negated standardized sum of the four
features with no learned parameters, reaches 0.673, slightly
above the trained gradient-boosted classifier at 0.645, and this
ordering is not a protocol artifact. Evaluating the threshold-only
variant under the same grouped five-fold cross-validation, with
the per-feature mean and standard deviation fit on the training
folds only, gives the same 0.673 [0.645,0.700] as fitting them
on the whole set, so there is no standardization leakage and
both estimates are out-of-fold. On a dataset of a few hundred
responses the tree ensemble overfits slightly relative to the zero-
parameter monotone sum, so the simpler detector generalizes
at least as well while requiring no labeled data. We therefore
use the threshold-only variant as the default GASP detector and
report the trained classifier for comparison. Adding perplexity
and length to the trained classifier reaches 0.678, within the
confidence interval of the threshold-only variant.
Page 15 of 23

## Page 16

Research Article
Bouke, 2026
Figure 6: Distribution of grounding sensitivity, measured as the maximum log-likelihood drop on context removal, for grounded versus
hallucinated sentences, shown for each of the three scorers.
Figure 7: Classifier feature importance (gain) at the span level, for the three scorers, with the four grounding features in navy and the perplexity
and length baselines in amber. No single feature dominates. The grounding features carry comparable gain, with length the weakest on each
scorer.
Table 12: Span-level comparison against the entailment and self-
consistency baselines, Qwen2.5-1.5B scorer, with 95% bootstrap confi-
dence intervals. Bold marks the recommended default detector, not the
maximum, since the chunk-level NLI verifier and the combined trained
set fall within its confidence interval.
Method
Span AUC
Perplexity
0.565 [0.532, 0.599]
Length
0.548 [0.512, 0.582]
NLI, small, whole context
0.532
NLI, large, whole context
0.605 [0.574, 0.637]
NLI, large, max-chunk
0.677 [0.649, 0.704]
SelfCheckGPT-NLI (N=4)
0.486 [0.446, 0.525]
GASP-threshold (default)
0.673 [0.645, 0.700]
GASP-trained
0.645 [0.615, 0.675]
GASP+base (trained)
0.678 [0.648, 0.707]
8.3 Ablation and sensitivity
Table 13 ablates the four features on the Qwen2.5-1.5B scorer
at the span level. No single feature matches the full set, and the
two views contribute in different proportions. The no-context
and divergence pairs are the strongest groups, and the leave-
one-out divergence is the weakest single feature, yet removing
it lowers the full-set score, so all four carry non-redundant infor-
mation. At the response level the balance shifts further toward
the whole-context view, where the no-context divergence alone
reaches 0.727 on the 1.5B scorer, close to the full-set 0.726,
while the localized leave-one-out view matters more at the span
level, where a single passage supports a sentence. The no-
context contrast, a single extra scoring pass, therefore carries
most of the discriminative power, especially at the response
level, and the K leave-one-out passes add a smaller amount of
complementary signal at the span level. The K+2 passes are
nonetheless justified because the leave-one-out passes produce
the attribution, the candidate supporting chunk for each sen-
tence, which the no-context contrast cannot provide, so they
buy the explanation rather than most of the detection. A deploy-
ment that needs only a detection score, and not the per-sentence
supporting passage, can run the two-pass full-versus-no-context
variant at most of the accuracy and a fraction of the cost.
Table 14 varies the two design choices a practitioner must set,
the chunk count K and the leave-one-out aggregation. The
span-level AUC changes by at most 0.03 across K ∈{3,5,10}
and across the maximum, mean, and top-two aggregations, and
every value overlaps the others within the confidence intervals,
so the method is not sensitive to these choices. We fixed K = 5
Page 16 of 23

## Page 17

Research Article
Bouke, 2026
Table 13: Feature ablation at the span level, Qwen2.5-1.5B scorer,
with 95% bootstrap confidence intervals. Single features (top) and
feature groups (bottom). All four features contribute, and the no-context
and divergence views carry most of the signal.
Feature subset
Span AUC
gap (no-context likelihood)
0.620 [0.588, 0.651]
jsd∅(no-context divergence)
0.616 [0.583, 0.645]
drop (leave-one-out likelihood)
0.600 [0.568, 0.632]
jsdloo (leave-one-out divergence)
0.595 [0.563, 0.625]
no-context only (gap, jsd∅)
0.638 [0.610, 0.666]
leave-one-out only (drop, jsdloo)
0.616 [0.582, 0.648]
likelihood only (gap, drop)
0.636 [0.607, 0.664]
divergence only (jsd∅, jsdloo)
0.640 [0.609, 0.668]
all four (GASP)
0.645 [0.615, 0.675]
Table 14: Sensitivity to the chunk count K and to the leave-one-out
aggregation, span level, Qwen2.5-1.5B, with 95% bootstrap confidence
intervals. The signal is stable across both choices.
Setting
Span AUC
K = 3 chunks
0.670 [0.639, 0.699]
K = 5 chunks (default)
0.645 [0.615, 0.675]
K = 10 chunks
0.640 [0.611, 0.670]
aggregation: max (default)
0.645 [0.615, 0.675]
aggregation: mean
0.625 [0.595, 0.655]
aggregation: top-2 sum
0.638 [0.606, 0.669]
before running any experiment, as a balance of localization and
cost, and report K ∈{3,10} only as a post-hoc sensitivity check,
so no value of K was selected on the test data. Coarser chunking
with K = 3 is marginally best, consistent with a sentence leaning
on a small span of context, but the difference is within noise.
8.4 Effect of truncation
Because a 6 gigabyte GPU bounds the context to 700 tokens and
the answer to 200, we quantify what truncation removes. About
42% of contexts on the Qwen scorers and 48% on SmolLM2
exceed 700 tokens and are cut, and about 25 to 28% of answers
exceed 200 tokens. The labels are largely preserved, since 90 to
92% of annotated hallucination spans start within the kept an-
swer, so most annotated spans remain scorable. Context trunca-
tion, however, can remove the passage that supports a grounded
sentence, which would lower that sentence’s measured sen-
sitivity and push it toward the hallucinated side. Truncation
therefore likely introduces noise and depresses sensitivity esti-
mates, which would make the reported numbers conservative,
but they are not a strict lower bound, since cutting the context
can also alter the effective grounding contract the scorer sees,
and a sentence that is supported in the full context may become
genuinely unsupported relative to the shortened input. A larger
context window is the most direct way to remove this confound.
8.5 Explanation quality
GASP returns, for each sentence, the chunk whose removal
most reduces its likelihood as a candidate supporting passage.
RAGTruth annotates hallucinated spans but not the gold support-
ing passage for grounded ones, so we cannot compute a gold hit
rate and instead run a proxy on the 2,273 grounded sentences of
the Qwen2.5-1.5B scorer, using the large NLI cross-encoder as
an independent notion of support. The attributed chunk entails
its sentence with mean probability 0.50, far above a random
chunk at 0.05 and the highest lexical-overlap chunk at 0.07,
with paired Wilcoxon p < 10−240 in both comparisons, so the
attribution tracks entailment rather than surface overlap. It is
also close to the best available, since the single most entailing
of the five chunks reaches 0.56 on average. In 69% of grounded
sentences the drop-attributed chunk is exactly the chunk the
NLI model ranks as most entailing, against a 20% chance rate
for five chunks. The best-chunk entailment is also higher for
grounded sentences than for hallucinated ones, 0.56 against
0.28 with p < 10−23, consistent with grounded sentences hav-
ing a genuine supporting passage and hallucinated ones not.
The Wilcoxon tests above are over the 2,273 grounded sen-
tences, which are not independent, since several come from
one response, so we also aggregate to the 399 responses that
contain a grounded sentence and repeat the test on per-response
means. The pattern holds, with attributed 0.50 against random
0.05 and lexical 0.07, a per-response agreement of 69%, and
paired Wilcoxon p < 10−62 against both controls, so the effect
is not an artifact of within-response dependence. These are
proxy results, since the reference is an NLI model rather than a
human evidence alignment, so we frame the returned chunk as
a candidate supporting passage rather than a verified citation. A
small human study that labels each attribution as correct, partial,
topical, or incorrect support would strengthen the claim further
and is left to future work.
8.6 Qualitative case study
Table 15 traces the method on two spans from the same Data2txt
overview task, both scored by Qwen2.5-1.5B, to show the sig-
nal behind the aggregate numbers. The first span is grounded
and the second is unsupported by the retrieved context. In the
grounded case the model reports that popular dishes include
sizzling rice soup and imperial shrimp, a detail carried by a sin-
gle customer review that names those exact dishes. Removing
that one chunk lowers the span log-likelihood by 3.55 nats and
moves its predictive distribution by a Jensen-Shannon diver-
gence of 0.37, the standardized grounding-sensitivity score is
+13.1, and the same leave-one-out step that produces the score
returns that review as the supporting chunk. In the unsupported
case the model asserts vegetarian, gluten-free, and vegan op-
tions that appear nowhere in the business record or its reviews.
The span counts as a hallucination because it is unsupported by
the retrieved evidence rather than false in the world, following
the grounding notion set out in Section 3.1. Here removing
any chunk changes the log-likelihood by at most 0.03 nats, the
largest divergence is 0.01, and the score is −5.9, so the span is
flagged. The two spans show the behavior the detector depends
on, since a grounded span leans on specific evidence and reacts
when that evidence is removed, whereas an unsupported span
is produced from the model prior and stays almost fixed as the
Page 17 of 23

## Page 18

Research Article
Bouke, 2026
Table 15: Qualitative case study of two spans from the same Data2txt task in RAGTruth, scored by Qwen2.5-1.5B. The grounded span reacts
sharply when its supporting chunk is removed, while the unsupported span is almost invariant to the context. All values are the actual per-span
GASP features, with gaps and drops in nats.
Supported span
Unsupported span
Query
“Write an objective overview about the following
local business based only on the provided structured
data... Don’t make up information.”
The same instruction, applied to a different business.
Generated span
“Some popular dishes include sizzling rice soup and
imperial shrimp.”
“The menu offers a variety of options...
such as
vegetarian, gluten-free, and vegan choices.”
Gold label
Grounded
Hallucinated (baseless)
Most influential chunk
Customer review: “... I personally really love the
sizzling rice soup and imperial shrimp.”
Customer review: “... a really good bison burger...
cheese pizza... French fries were way too salty”,
with no mention of dietary options.
Gap,
all context re-
moved
+3.52
+0.09
Max chunk drop, leave-
one-out
+3.55
+0.03
Max chunk JSD
0.37
0.01
Grounding-sensitivity
score
+13.1
−5.9
GASP outcome
Supported. The review chunk is returned as the evi-
dence, and the span restates it, so removing that chunk
collapses the likelihood.
Flagged as a likely hallucination. The span states
details not supported by the retrieved evidence, so
removing any chunk barely changes the likelihood.
context changes.
The same two spans make the reason for the dynamical-systems
framing concrete. Figure 8 plots the per-token log-probability
of each span with the retrieved context and without it. For the
grounded span, the content words such as “soup”, “imperial”,
and “shrimp” are near-certain with the context and fall to log-
probabilities between about −4 and −8 once it is removed,
so the span is a high-probability continuation only under the
context. In the language of Section 5.1, the answer sits on
the attractor induced by the retrieved evidence, and removing
that evidence moves the attractor out from under it. For the
hallucinated span, the two curves lie almost on top of each other,
so the unsupported options are equally probable with or without
the context, which places the span on the model’s context-free
prior rather than on the context attractor. Grounding sensitivity
is the size of this gap aggregated over the span, which is why we
read grounding as movement of a context-conditioned measure
rather than as a surface property of the text.
8.7 Attractor estimation and controls
Section 5.1 framed decoding under a context as a stable attractor
and derived, in Proposition 1, that a context edit moves that at-
tractor by an amount the detector reads. We test the observable
consequences on a balanced sample of 200 responses scored by
Qwen2.5-1.5B, extracting the answer-token trajectory and esti-
mating its correlation dimension by the Grassberger-Procaccia
method with a Euclidean metric and a scaling region between
the 5th and 60th percentiles of pairwise distances. First, the
trajectory is low-dimensional and this is stable across depth,
with correlation dimension 8.3, 9.8, and 8.0 at an early, middle,
and late layer in a hidden space of dimension 1536. That the
estimator is meaningful rather than always small is confirmed
by an iid-Gaussian cloud of the same size, whose estimated
dimension is 74. The no-context trajectory has a similar dimen-
sion of 9.3, so low-dimensionality reflects the general geometry
of decoding and is not specific to the context attractor. Second,
and more discriminating, removing the most influential chunk
by likelihood moves the trajectory more than every control, with
a mean movement of 8.4 against 3.3 for a random chunk, 3.7
for a length-matched chunk, 4.2 for the highest lexical-overlap
chunk, and 4.1 for the first chunk, so the movement is not an
artifact of chunk length, lexical overlap, or position. The paired
Wilcoxon tests over the 200 responses give p < 10−25 against
each control. Figure 9 illustrates this on one representative
response, where removing the most influential chunk pulls the
answer-token trajectory away from its full-context path while a
length-matched removal leaves it almost unchanged. The move-
ment also correlates with the likelihood feature the detector
uses, Spearman 0.71 (p < 10−30) at the response level. The
hidden-state movement does not, however, by itself separate
grounded from hallucinated sentences at this sample size, 8.55
against 8.19 with p = 0.15, so the discrimination the detector
achieves comes from the likelihood and divergence readouts
rather than from the raw magnitude of the trajectory movement.
Taken together, the measurements support the interpretation
that a specific context edit moves a low-dimensional decod-
ing trajectory, and they do so under controls that rule out the
simplest confounds, but they do not prove the contraction as-
sumption and they do not show that trajectory geometry alone
is a detector.
8.8 Summary of findings
The results support five main findings. Grounding sensitivity
ranks hallucinated content above grounded content at both gran-
ularities and on all three scorers, at a moderate absolute level
that is clearly above perplexity, length, whole-context entail-
Page 18 of 23

## Page 19

Research Article
Bouke, 2026
Figure 8: Per-token log-probability of the two case-study spans, with and without the retrieved context, under Qwen2.5-1.5B. (a) The grounded
span collapses when the context is removed. (b) The hallucinated span is almost unchanged. The mean gap between the curves, +3.52 against
+0.09 nats, is the grounding-sensitivity signal.
Figure 9: Answer-token hidden-state trajectory at the middle layer for a representative grounded response, under the full context, removal of the
most influential chunk, and removal of a length-matched control (Qwen2.5-1.5B). (a) Two-dimensional PCA projection. Removing the influential
chunk pulls the trajectory away from the full-context path, while the control removal stays close. (b) Per-token shift from the full-context trajectory,
larger for the influential-chunk removal, with mean movement 9.1 against 3.7.
ment, and a self-consistency baseline, and competitive with
a well-configured chunk-level entailment verifier at the span
level while needing no separate verifier and while also provid-
ing a response-level score and an attribution. A training-free
threshold on the standardized features matches the trained clas-
sifier, so no labeled data is required. The signal is consistent
across the three tested scorers, whose AUC agrees within the
confidence intervals, and the grounding and baseline signals are
complementary at the span level while grounding sensitivity
alone carries the response-level signal. The signal transfers to
a second summarization benchmark, TofuEval, but a probe on
RAGBench shows it is not universal, since it is strongest for
context-constructed outputs such as summarization and data-to-
text and weak on short-answer QA, where parametric knowl-
edge can substitute for retrieved evidence. The hidden-state
measurements are consistent with a low-dimensional decoding
trajectory that moves more under the most influential edit than
under length, lexical, and position controls, though they do
not show that trajectory geometry alone separates grounded
from hallucinated sentences. Together these support the central
claim that grounding is best tested relationally, by perturbing
the evidence, rather than read from intrinsic properties of the
text.
9 Discussion
Page 19 of 23

## Page 20

Research Article
Bouke, 2026
9.1 Claim hierarchy
This section separates what the paper proves, what it measures,
and what it conjectures. The proven part is narrow and condi-
tional. Under the average-contraction assumption, Proposition 1
bounds the movement of the conditional generation attractor
by the JSD the detector computes, and Propositions 2 and 3 fix
the limiting behavior and the scale of the features. These are
statements about an idealized RNIFS, not about a specific trans-
former, and we do not claim the assumption holds exactly for
the models we use. The measured part is the substance of the
paper. On a span-annotated benchmark, under a leakage-clean
protocol with bootstrap confidence intervals and paired sig-
nificance tests, grounding sensitivity ranks hallucinated spans
above grounded ones better than perplexity, length, and whole-
context entailment, and it does so consistently across three scor-
ers that span two model families. The absolute level is moderate
rather than decisive. The conjectured part is that the measured
signal reflects the attractor movement the theory describes. The
correlation-dimension and trajectory-movement estimates in
Section 8.7 support this reading but do not establish it, and
a signed, direction-aware variant is a hypothesis we leave to
future work. A reader should take the empirical ranking claims
as the load-bearing ones, the theory as the reason the features
take the form they do, and the dynamical-systems language as
an interpretation that the measurements make plausible rather
than prove. The scope of the empirical claim is specific. GASP
should be understood as a detector of dependence on retrieved
evidence, not a universal detector of factual correctness, which
is why it helps most where the answer is constructed from the
context and adds little on short-answer questions the model can
answer from parametric knowledge.
9.2 Interpretation of the mechanism
Grounding sensitivity measures a relational, counterfactual
property, how much a span depends on its evidence, not an
intrinsic property of the text. This is why it separates hallu-
cinated from grounded spans where fluency-based signals do
not. It asks how the answer would have changed had a passage
been absent, in the same spirit as influence analysis, and the
answer to that question is what grounding means. A surface-
overlap comparison would instead conflate topical similarity
with grounding, because an unsupported sentence on the same
topic shares many tokens with the context while depending
on none of them. This is also why the signal is largely in-
sensitive to how fluently the answer is written, since a fluent
and a clumsy paraphrase of the same passage both collapse in
probability when the passage is removed, and a fluent and a
clumsy unsupported sentence both fail to react, so the fluency
axis along which intrinsic detectors vary is orthogonal to the
dependence axis the method measures.
9.3 Per-type behavior
We expected the baseless type to be the easier case, but evident-
conflict spans are detected slightly and consistently better. A
plausible reason is that a conflicting span interacts strongly with
the passage it contradicts, so its likelihood is sensitive to that
passage’s removal, while some baseless additions are topically
plausible and only weakly sensitive to any single unit. This
suggests a signed variant that records whether removal raises or
lowers the likelihood, since reliance lowers it and contradiction
may raise it once the conflicting evidence is gone. The current
magnitude feature collapses these cases, which we leave as
future work.
9.4 Relation to faithfulness evaluation
The summarization literature has long measured faithfulness,
the requirement that a summary assert only what its source
supports, and it established that faithfulness is distinct from
fluency and is best judged at a fine granularity. Grounding
sensitivity operationalizes the same requirement for retrieval-
augmented answers, but it differs from the standard faithfulness
metrics in three ways that matter for deployment. It is computed
from a probabilistic scorer rather than by an external judge, so
there is no separate verifier to maintain or to match to the
domain. It is defined at the span level by construction, so it
does not require a separate decomposition of the answer into
atomic units before scoring. And it produces, for each span, the
passage that supports it, which a faithfulness score alone does
not. Where decomposition-based evaluation multiplies verifier
calls across atomic facts, the present method amortizes a fixed
number of scoring passes across the whole answer and reads
every sentence from the same passes, which is what makes it
cheap enough to run on every answer rather than as an offline
evaluation.
9.5 Deployment and use
The detector fits an explainable guardrail that slots into a RAG
pipeline after generation, since the pipeline already holds the
query, the chunks, and the answer. Given the moderate absolute
performance, its right role is a triage and review-prioritization
signal, an assistive guardrail rather than a standalone safety
filter that gates content without human oversight. In that role a
deployment can highlight low-sensitivity sentences for review,
attach the supporting passage to each grounded sentence as
an inline citation, or, only in a lower-stakes setting, hold back
sentences whose grounding sensitivity falls below a chosen
threshold. The AUC is threshold-free, but a deployment must
choose an operating point from the cost of review and from
a representative class prior, since a live stream is dominated
by grounded content, which is why we report the full ROC
rather than a single threshold-dependent score. More broadly,
the method is detection, and it is complementary to improving
retrieval, because even with strong retrieval a generator can
ignore or contradict a supplied passage. The two address differ-
ent failure points, since better retrieval reduces the chance that
the model lacks the evidence it needs, while detection catches
the cases where the model fails to use the evidence it has. The
method also yields a retrieval-side benefit indirectly, because
the supporting passage it returns for each grounded sentence is
exactly the attribution a citation interface would display, which
can be reused to surface evidence to the user. Because the scorer
is independent of the system under audit, the guardrail can be
deployed without access to the internals of the generator that
produced the answers, which matters when that generator is
hosted or proprietary.
9.6 Absolute performance
The absolute scores are moderate, and detection from the an-
swer and its context alone is hard, because some grounded
Page 20 of 23

## Page 21

Research Article
Bouke, 2026
sentences restate widely known facts that a model would assign
high probability without any passage, and some hallucinations
inherit enough topical context to behave like grounded text.
This level is reached with a small scorer, a compact feature
family, and no task-specific training. The comparison that mat-
ters is relative, and grounding sensitivity sits well above the
intrinsic and whole-context baselines under the same protocol,
is competitive with a well-configured chunk-level entailment
verifier, and additionally localizes and explains its decisions.
The path to higher absolute numbers, through stronger scorers,
richer aggregation, and complementary signals, is open and
does not alter the central finding that grounding must be tested
by perturbing the evidence.
9.7 Cost and relation to influence
The dominant cost is the K +2 scoring passes per response in
Table 3. Because the answer is fixed, these are scoring passes
rather than generation, far cheaper than decoding the answer,
and they parallelize across conditions and responses. The linear
factor can be reduced by coarser chunking or by restricting the
leave-one-out set to the highest-ranked retrieved units. Concep-
tually the measure is close to leave-one-out influence estimation.
As an alternative, reading attention as a proxy for dependence
is cheaper, but attention indicates where the model looks rather
than whether the output would change if a unit were gone, so
we prefer the counterfactual measure and view attention as a
cost reduction rather than the primary signal.
9.8 Failure modes
The method has two characteristic failure modes. A baseless
addition that is topically plausible may draw weak support from
several units at once, so no single removal produces a large
drop and the span looks grounded, which is consistent with
the slightly lower detection of the baseless type. A grounded
span whose supporting passage is split across two chunks may
have its support divided, so the maximum single-chunk drop
understates its dependence. Both point to the same refinement, a
chunking that aligns with the unit of support, and to combining
the maximum with a top-few aggregation. The chunk count
itself controls a trade-off, since very coarse chunks make almost
every sentence react and narrow the contrast, while very fine
chunks raise the cost and risk splitting a passage, so the useful
regime is one where a chunk corresponds to a unit of evidence.
In the range we tested, K ∈{3,5,10} in Table 14, detection is
insensitive to the choice, with coarser chunks marginally ahead,
so within a reasonable range the trade-off is mild.
9.9 Threat model
In the intended setting the threat is an honest generator that
occasionally produces ungrounded content, not an adversary
that crafts answers to evade the detector. Under that assump-
tion the signal is well matched, since ordinary hallucinations
are produced from the prior and are insensitive to context by
construction. An adaptive adversary could in principle phrase
an unsupported answer so that it borrows just enough from a
passage to react under removal, which would raise its ground-
ing sensitivity and evade the detector. Defending against such
adaptive evasion is out of scope here, and it would require the
usual robustness treatment of adaptive attacks, which we note
as a direction rather than a claim. Evasion is not free for the
adversary, however, since making an unsupported answer re-
act to a passage means tying it to that passage content, which
pulls the statement toward being grounded in it, so the space of
unsupported answers that both evade the detector and remain
ungrounded is narrower than the space of unsupported answers
at large. The detector therefore raises the cost of producing
fluent, unsupported content that also looks grounded under per-
turbation, even if it does not eliminate it, and combining it with
complementary signals would further constrain the adversary.
10 Threats to Validity
The construct validity of the labels depends on the human an-
notations of the benchmark, which we take as ground truth,
and annotation noise places a ceiling on measurable perfor-
mance. The internal validity of the comparison depends on the
leakage-clean protocol, which is easy to do incorrectly, since
pooling all spans and running an ordinary cross-validation lets
a classifier train on some sentences of an answer and test on
others, and the resulting inflation is not small. Grouping folds
by answer removes this path. The external validity rests on two
benchmarks in two domains, RAGTruth and TofuEval, which
is stronger than one but still a limited sample. The absolute
numbers may therefore differ elsewhere, and short-answer ques-
tion answering would need a different segmentation. The two
task types were chosen because their answers are long enough
for stable per-sentence estimation, and the method is expected
to be weaker where answers are very short, since a sentence
of a few tokens gives a noisy estimate of dependence. The
three scorers span two model families, so the robustness we
observe is across scale and across family, not within a sin-
gle lineage. Three models are still a small sample, however.
Beyond the scorers, benchmark coverage is improved but the
method has a scope boundary that the coverage makes explicit.
Grounding sensitivity shows clear, significant gains on two
summarization-faithfulness benchmarks, RAGTruth and the
MeetingBank portion of TofuEval in Section 7.6, which span
two domains and two annotation schemes, but on the multi-
domain short-answer QA of RAGBench in Section 7.7 it is only
comparable to perplexity, which bounds the empirical claims to
the summarization setting. In addition, all three benchmarks are
English, which bounds the external validity of these results to
English. The class-balanced sampling departs from the natural
prior, which suits a threshold-free ranking metric but means
the numbers are not operating-point estimates. For the base-
lines, the entailment comparison uses both a compact and a
large verifier and a self-consistency baseline, so it is not resting
on one weak verifier. Conclusion validity is supported by the
consistency of the effect across three independent scorers and
across the response and span levels, together with bootstrap
confidence intervals and paired significance tests that place the
grounding gap over perplexity outside the interval on every
scorer. This makes it unlikely that the result is an artifact of
a single configuration, although a larger study with multiple
benchmarks would strengthen the statistical claims further. We
also note that the class-balanced construction is a deliberate
choice for measuring ranking quality and is not a claim about
prevalence, so the numbers should not be read as the rate at
which a live system hallucinates.
Page 21 of 23

## Page 22

Research Article
Bouke, 2026
11
Ethical Considerations
A hallucination detector is a safety tool, and its limitations carry
ethical weight. The detector is imperfect, so it must support
human review rather than replace it, and an automated filter that
silently drops flagged content could suppress correct statements
that score low. Over-reliance is a second risk, since a detector
that is usually right can train its users to stop checking, so a
sentence that slips through is trusted more, not less. Separately,
the method does not generate content and adds no data collec-
tion beyond the public benchmark, and the supporting-passage
explanation is intended to keep a person in the loop. We rec-
ommend deploying it as an assistive guardrail that surfaces and
explains, with final judgment left to a person in high-stakes
settings, and communicating its confidence and known failure
modes alongside its flags. Transparency about scope is part
of responsible use. The detector judges grounding in the re-
trieved evidence, not truth in the world, so a statement that is
faithfully grounded in a passage that is itself wrong will not
be flagged, and users should understand that the tool checks
consistency with the provided sources rather than correctness.
Presenting the supporting passage alongside each decision helps
here, because it lets a reviewer judge both whether the sentence
is grounded and whether the source it relies on is trustworthy,
which is a judgment the detector does not make on its own.
12
Conclusion
We presented GASP, a span-level detector for hallucination in
RAG that scores each answer sentence by its grounding sen-
sitivity, how strongly its likelihood depends on the retrieved
evidence. We framed decoding under a context as a context-
conditioned RNIFS whose invariant measure is the attractor
of grounded continuations, and we measured the sensitivity
of that attractor to context perturbation using log-likelihood
drops and JSD under leave-one-context-out removal. On two
span-annotated benchmarks in two domains, RAGTruth and the
MeetingBank portion of TofuEval, with three instruction-tuned
scorers spanning two model families and a leakage-clean eval-
uation, grounding sensitivity outperformed perplexity, with a
significant gap under a paired bootstrap test on all three scor-
ers, and outranked length, whole-context entailment, and a
self-consistency baseline, while remaining competitive with a
well-configured chunk-level entailment verifier at the span level.
It was consistent across the three tested scorers and produced
a candidate supporting passage for each flag at no extra cost.
A probe on the multi-domain short-answer QA of RAGBench
marked the boundary of the approach, since there grounding
sensitivity was only comparable to perplexity, which places
the method in settings where the answer must be constructed
from the retrieved context rather than recalled from parametric
knowledge. The approach reframes hallucination as a failure
of dependence on evidence rather than a property of the text,
and it offers an explainable, inexpensive guardrail for retrieval-
augmented systems. More broadly, it argues for measuring
grounding through controlled perturbation of the evidence, a
principle that extends beyond the specific features used here.
The qualitative finding, that the reaction to evidence removal
separates grounded from ungrounded content where intrinsic
fluency does not, follows from what grounding means rather
than from the particular instantiation we evaluated.
13 Limitations and Future Work
The absolute AUC is moderate, in the range from 0.64 to 0.75,
which is expected for hard hallucination detection with small
models and a compact feature family, and which marks the
method as a signal that beats standard baselines rather than
a finished product. Larger and stronger instruction-following
scorers may raise the ceiling, and the same perturbation machin-
ery applies unchanged. Additional RAG domains and bench-
marks would test external validity, and the method is language-
agnostic in principle, so an evaluation in other languages, in-
cluding low-resource settings where annotated data is scarce,
would test whether grounding sensitivity transfers. The empir-
ical comparison covers the baseline families that can be run
under a single leakage-clean protocol, and it does not include a
head-to-head with the white-box mechanistic detectors or with
a supervised RAGTruth-trained judge, which occupy different
operating regimes. A unified benchmark that evaluates black-
box and white-box detectors under one protocol would make
these comparisons direct and is left to future work. The per-type
finding motivates separating the direction of sensitivity from
its magnitude. Multi-passage support, where a claim rests on
the combination of two or more passages that the single-chunk
maximum cannot capture, motivates a small-subset removal or
a top-few aggregation. Cheaper approximations of the pertur-
bation, for example attention-based or gradient-based proxies,
would reduce the linear cost in the number of chunks. An adap-
tive chunking that follows retriever passage boundaries would
align the unit of removal with the unit of evidence and is likely
to sharpen both detection and localization. Finally, the quantita-
tive link between the measured divergence and the movement
of the invariant measure, which Proposition 1 establishes under
an average-contraction assumption and Section 8.7 tests on the
hidden states, could be tightened into a model-specific bound
by estimating the contraction and spread constants of a given
network directly, which would replace the qualitative account
with a numerical guarantee.
Declarations
• Funding: Not applicable.
• Conflict of Interest: The author declares no competing
interests.
• Availability of Data and Materials: All three bench-
marks are publicly available (RAGTruth, the Meeting-
Bank portion of TofuEval, and RAGBench). The full
implementation and evaluation pipeline, including the
scoring, feature extraction, leakage-clean evaluation, and
the scripts that generate every table and figure, is avail-
able as open-source code at https://github.com/
drbouke/GASP.
• Ethics Approval: Not applicable.
Page 22 of 23

## Page 23

Research Article
Bouke, 2026
Author Biography
Mohamed Aly Bouke is a Postdoc-
toral Researcher at Multimedia University,
Malaysia, and a Senior Member of IEEE.
He received his PhD in Information Secu-
rity from Universiti Putra Malaysia. He
serves as an Associate Editor for e-Prime
– Nexus of Electrical, Electronic, and In-
telligent Engineering. His research inter-
ests include cybersecurity, artificial intel-
ligence, cryptography, and mathematical
modeling.
References
[1]
P. Lewis, E. Perez, A. Piktus, F. Petroni, et al., “Retrieval-
augmented generation for knowledge-intensive NLP
tasks,” in Advances in Neural Information Processing
Systems (NeurIPS), 2020.
[2]
Z. Ji, N. Lee, R. Frieske, T. Yu, et al., “Survey of halluci-
nation in natural language generation,” ACM Computing
Surveys, vol. 55, no. 12, pp. 1–38, 2023.
[3]
L. Huang, W. Yu, W. Ma, W. Zhong, et al., “A survey
on hallucination in large language models: Principles,
taxonomy, challenges, and open questions,” ACM Trans-
actions on Information Systems, 2025.
[4]
S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, “Detecting
hallucinations in large language models using semantic
entropy,” Nature, vol. 630, pp. 625–630, 2024.
[5]
C. Chen et al., “INSIDE: LLMs’ internal states retain
the power of hallucination detection,” in International
Conference on Learning Representations (ICLR), 2024.
[6]
G. Sriramanan, S. Bharti, V. S. Sadasivan, S. Saha, P.
Kattakinda, and S. Feizi, “LLM-Check: Investigating
detection of hallucinations in large language models,”
in Advances in Neural Information Processing Systems
(NeurIPS), 2024.
[7]
P. Manakul, A. Liusie, and M. J. F. Gales, “SelfCheck-
GPT: Zero-resource black-box hallucination detection
for generative large language models,” in Proceedings of
the Conference on Empirical Methods in Natural Lan-
guage Processing (EMNLP), 2023.
[8]
S. Min, K. Krishna, X. Lyu, M. Lewis, et al., “FActScore:
Fine-grained atomic evaluation of factual precision in
long form text generation,” in Proceedings of the Con-
ference on Empirical Methods in Natural Language Pro-
cessing (EMNLP), 2023.
[9]
B. Cohen-Wang, H. Shah, K. Georgiev, and A. Madry,
“ContextCite: Attributing model generation to context,”
in Advances in Neural Information Processing Systems
(NeurIPS), 2024.
[10]
Y.-S. Chuang, L. Qiu, C.-Y. Hsieh, R. Krishna, Y. Kim,
and J. Glass, “Lookback lens: Detecting and mitigating
contextual hallucinations in large language models using
only attention maps,” in Proceedings of the Conference
on Empirical Methods in Natural Language Processing
(EMNLP), 2024.
[11]
Z. Sun et al., “ReDeEP: Detecting hallucination in
retrieval-augmented generation via mechanistic inter-
pretability,” in International Conference on Learning
Representations (ICLR), 2025.
[12]
S. Yeh, Y. Li, and T. Mallick, “LUMINA: Detecting
hallucinations in RAG systems with context-knowledge
signals,” in International Conference on Learning Repre-
sentations (ICLR), 2026.
[13]
C. Niu, Y. Wu, J. Zhu, S. Xu, et al., “RAGTruth: A
hallucination corpus for developing trustworthy retrieval-
augmented language models,” in Proceedings of the An-
nual Meeting of the Association for Computational Lin-
guistics (ACL), 2024.
[14]
S. Es, J. James, L. Espinosa-Anke, and S. Schockaert,
“RAGAs: Automated evaluation of retrieval augmented
generation,” in Proceedings of the European Chapter of
the Association for Computational Linguistics (EACL):
System Demonstrations, 2024.
[15]
Á. Kovács and G. Recski, LettuceDetect: A hallucina-
tion detection framework for RAG applications, arXiv
preprint arXiv:2502.17125, 2025.
[16]
M. A. Bouke, “Fractal attractors in random nonlinear it-
erated function systems: Existence, stability, and dimen-
sional properties,” Chaos, Solitons & Fractals, vol. 207,
p. 117 996, 2026.
[17]
G. Ke, Q. Meng, T. Finley, T. Wang, et al., “Light-
GBM: A highly efficient gradient boosting decision tree,”
Advances in Neural Information Processing Systems
(NeurIPS), 2017.
[18]
L. Tang, I. Shalyminov, A. W.-m. Wang, J. Fang, et al.,
“TofuEval: Evaluating hallucinations of large language
models on topic-focused dialogue summarization,” in
Proceedings of the Conference of the North American
Chapter of the Association for Computational Linguistics
(NAACL), 2024.
[19]
Y. Hu, T. Ganter, H. Deilamsalehy, F. Dernoncourt,
H. Foroosh, and F. Liu, “MeetingBank: A benchmark
dataset for meeting summarization,” in Proceedings of
the Annual Meeting of the Association for Computational
Linguistics (ACL), 2023.
[20]
R. Friel, M. Belyi, and A. Sanyal, “RAGBench: Explain-
able benchmark for retrieval-augmented generation sys-
tems,” in arXiv preprint arXiv:2407.11005, 2024.
[21]
Qwen Team, “Qwen2.5 technical report,” Alibaba Group,
Tech. Rep., 2025, arXiv preprint.
[22]
L. B. Allal, A. Lozhkov, E. Bakouch, et al., “SmolLM2:
When smol goes big – data-centric training of a small lan-
guage model,” arXiv preprint arXiv:2502.02737, 2025.
Page 23 of 23
