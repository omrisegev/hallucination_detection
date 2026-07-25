---
source_pdf: papers/FUSE - Ensembling Verifiers with Zero Labeled Data.pdf
slug: fuse-ensembling-verifiers-with-zero-labeled-data
pages: 24
extracted_on: 2026-07-13
---

# FUSE - Ensembling Verifiers with Zero Labeled Data

## Page 1

FUSE: Ensembling Verifiers with Zero Labeled Data
Joonhyuk Lee * 1 Virginia Ma * 1 Sarah Zhao * 1 Yash Nair 1 Asher Spector 1 Regev Cohen 2
Emmanuel Cand`es 1
Abstract
Verification of model outputs is rapidly emerging
as a key primitive for both training and real-world
deployment of large language models (LLMs).
In practice, this often involves using imperfect
LLM judges and reward models since ground
truth acquisition can be time-consuming and ex-
pensive. We introduce Fully Unsupervised Score
Ensembling (FUSE), a method for improving ver-
ification quality by ensembling verifiers without
access to ground truth correctness labels. The
key idea behind FUSE is to control conditional
dependencies between verifiers in a manner that
improves the unsupervised performance of a class
of spectral algorithms from the ensembling lit-
erature. Despite requiring zero ground truth la-
bels, FUSE typically matches or improves upon
semi-supervised alternatives in test-time scaling
experiments with diverse sets of generator models,
verifiers, and benchmarks. In particular, we vali-
date our method on both conventional academic
benchmarks such as GPQA Diamond and on fron-
tier, unsaturated benchmarks such as Humanity’s
Last Exam and IMO Shortlist questions.
1. Introduction
1.1. Motivation
Significant recent progress in the performance of large lan-
guage models (LLMs) has been driven by test-time scaling.
A key ingredient in many test-time scaling approaches is
Best-of-N (BoN) sampling, in which N responses are sam-
pled independently for any given query, and scores from a
verifier are used to select a single response to return (e.g.,
Lightman et al., 2023; Cobbe et al., 2021; Nakano et al.,
2021; Sun et al., 2024; Rakhsha et al., 2025; Di et al., 2025).
Such schemes, for instance, have been credited as a ma-
jor input for the gold-level performance of frontier LLMs
at the International Math Olympiad (Luong et al., 2025).
*Equal contribution 1Stanford University 2Google. Correspon-
dence to: Joonhyuk Lee <joonhyuk@stanford.edu>.
Preprint. April 21, 2026.
The verifiers used in BoN sampling are typically external
language models or reward models. While imperfect, such
models do not incur the time and monetary costs of acquir-
ing ground-truth labels from human experts. Notably, this
practical constraint extends beyond test-time scaling alone—
the pipeline of repeated sampling followed by imperfect
verification has also embedded itself into numerous tech-
niques for training language models, such as reinforcement
learning with rubric-based rewards (Gunjal et al., 2025) and
synthetic data selection (Liu et al., 2024).
With the imperfect nature of practical verifiers in mind, re-
cent work has proposed scaling verification (Zhao et al.,
2025; Saad-Falcon et al., 2025b), wherein scores for re-
sponses are computed by aggregating the outputs of multiple
verifiers. In the simplest case, this aggregation involves tak-
ing an average or majority-vote across verifiers (Verga et al.,
2024; Lifshitz et al., 2025). The performance of such rules
depends critically on the quality of the verifiers at hand. As
noted by Saad-Falcon et al. (2025b), when verifiers exhibit
disparate performance, average- and majority-vote-based
schemes may perform poorly because they treat scores from
unreliable verifiers on par with those given by their stronger
counterparts. Further complicating matters is that verifier
strength can be query-dependent. These nuances are dif-
ficult to adapt to, as by the previously cited constraints of
money and time, one often has zero ground-truth labels of
response correctness for any given query.
1.2. FUSE
To address the inherent challenges in constructing an en-
semble of verifiers in the absence of ground-truth labels, we
develop Fully Unsupervised Score Ensembling (FUSE), a
method which aggregates scores from multiple verifiers in
a data-adaptive manner to select the most promising LLM-
generated response at inference-time. Because FUSE learns
how to effectively ensemble verifiers from data, it gener-
ally performs better than (sometimes, significantly so) sim-
ple unweighted baselines. Despite using zero labeled data,
FUSE also performs competitively with prior methods that
do assume access to labeled data, and sometimes even out-
performs them. Figure 1 summarizes these findings in the
context of Best-of-N test-time scaling experiments on two
datasets considered in Saad-Falcon et al. (2025b) and the
1
arXiv:2604.18547v1  [stat.ML]  20 Apr 2026

## Page 2

FUSE: Ensembling Verifiers with Zero Labeled Data
IMO Shortlist subset of Google’s IMO AnswerBench (Lu-
ong et al., 2025).
On the methodological side, FUSE builds on the statistical
literature on crowd-sourcing and unsupervised ensemble
learning, in particular on the works of Parisi et al. (2014);
Jaffe et al. (2015). The methods proposed in these papers,
which consider unsupervised ensemble learning for binary
classification, impose various conditional independence as-
sumptions between classifiers. A key idea behind FUSE is
to mitigate the impact of violations of some of these assump-
tions, which we do not expect to hold in general for LLM
verifiers.
Two building blocks, visualized in Figure 2, underlie FUSE:
1. Under an assumption which we refer to as triplet con-
ditional independence (TCI), a spectral algorithm in
Jaffe et al. (2015) yields consistent estimates of verifier
quality. Rather than assuming TCI, we introduce an
empirical measure of TCI violations that can be com-
puted without labels. Using this statistic, FUSE learns
a transformation of raw verifier scores (Step 1 in Fig-
ure 2) for which applying the algorithm of Jaffe et al.
(2015) will yield reliable estimates (Figure 2, Step 2).
2. With estimates of verifier quality in hand, we form
pseudo-labels that can be used to optimize arbitrary
parametrized rules for aggregating verifier scores (e.g.
logistic regression) as visualized in Step 3 of Figure 2.
Our construction and subsequent use of pseudo-labels
allows us to avoid assuming joint conditional indepen-
dence (JCI)—a common assumption used in the crowd-
sourcing literature to ensemble classifiers—which is
empirically untestable and unlikely to hold in practice.
In summary, FUSE adaptively transforms verifier scores to
better satisfy a weaker TCI assumption while bypassing the
stronger JCI assumption. By extending Jaffe et al. (2015),
we also ensure that all steps in this procedure are compat-
ible with both real and discrete-valued verifiers. Before
providing further detail, we first fix notation and give a clear
problem statement.
1.3. Problem statement
Given a query q, we assume access to N LLM-generated
responses (r1, . . . , rN) and a collection of m verifiers
v1, . . . , vm that assign correctness scores to each query-
response pair. We treat the query as fixed (i.e., non-random),
while (r1, . . . , rN) are sampled i.i.d. from G(q), the distri-
bution of the generator model’s responses given the prompt
q.
We let yi := y(q, ri) ∈{±1} denote the ground-
truth correctness label of the ith response for the query:
yi = 1 if and only if ri is correct for q. Finally, letting
vi,j := vj(q, ri) be the score assigned by the jth verifier to
response ri, we write V := (vi,j)i,j for the N × m matrix
of verifier scores, and Vi• for the i-th row of V.
Our goal is to select, using the score matrix V alone, a re-
sponse ri⋆for which y(q, ri⋆) = 1. When multiple queries
q1, . . . , qL exist, our goal will be to select a correct response
for each question. In the latter case, we may pool infor-
mation from different score matrices V1, . . . , VL (‘batch-
ing’) or only use Vℓto select responses for qℓ(‘query-
conditional’). To minimize notational overhead, we restrict
our attention to the single-query case in subsequent sections.
2. Fully unsupervised score ensembling
2.1. MoM estimation of query-specific verifier qualities
The first step in FUSE is to estimate each verifier’s accuracy
conditioned on the query q using V. When verifiers satisfy
strong independence assumptions, we can do so by adapting
a method-of-moments (MoM) approach developed by Jaffe
et al. (2015) in the context of binary classification. In this
section, we provide necessary background for FUSE by
explaining this approach, and empirically justify the need
to look beyond it. For ease of exposition, we temporarily
assume, as in Jaffe et al. (2015), that verifiers output binary
{±1}-valued predictions. We later relax this assumption,
allowing FUSE to work with arbitrary combinations of
discrete and real-valued verifiers.
Because all outputs are binary, the quality of any given
verifier vj’s predictions on a response r for a query q is
determined by its sensitivity ψj and specificity ηj:
ψj := Pr∼G(q)(vj(q, r) = 1 | y(q, r) = 1)
ηj := Pr∼G(q)(vj(q, r) = −1 | y(q, r) = −1)
where, as a reminder, y(q, r) is the ground-truth correct-
ness label of the response for the query. Jaffe et al. (2015)
propose using the score matrix V to estimate the verifier
sensitivities ψ := (ψj)m
j=1 and specificities η := (ηj)m
j=1
under the following two assumptions:
Assumption 2.1 (Majority of classifiers are better than ran-
dom). Let πj := ψj+ηj
2
denote the balanced accuracy of
the jth verifier and π := (π1, . . . , πm) denote the vector of
balanced accuracies for query q. Then, more than m
2 of the
values {πj}m
j=1 are larger than 1/2.
Assumption 2.2 (Triplet conditional independence). Let-
ting r ∼G(q), each triplet of verifiers produces condition-
ally independent scores given the ground-truth label y(q, r)
(see Appendix A.1 for a precise statement).
Conditions like Assumption 2.1 are common in the crowd-
sourcing and ensemble learning literature (c.f., e.g., Dawid
& Skene, 1979; Kleindessner & Awasthi, 2018; Shaham
et al., 2016; Didwania et al., 2022; Ahsen et al., 2019)
2

## Page 3

FUSE: Ensembling Verifiers with Zero Labeled Data
0
10
20
30
22.1
21.8
7.2
15.3
0
5
10
15
20
25
21.5
20.3
4.5
17.1
0
10
20
30
10.5
8.9
4.4
10.5
    Accuracy Improvement over Pass@1 (%)
GPQA Diamond (k = 100)
Pass@1 = 42.3%
MMLU-Pro (k = 100)
Pass@1 = 69.9%
IMO Shortlist (k = 50)
Pass@1 = 53.3%
FUSE
Weaver
Majority Vote
Naive Ensemble
Pass@k (Oracle)
Figure 1. BoN accuracy of our method versus that of a leading semi-supervised alternative (WEAVER, by Saad-Falcon et al. (2025b))
and unsupervised baselines of naive ensemble and majority vote. All bars are re-scaled to depict improvement over Pass@1, which is
the accuracy of a random selection rule. The black dotted Pass@k line denotes the maximum possible accuracy improvement for any
selection method. Despite being unsupervised, FUSE is competitive with WEAVER and decisively beats naive ensemble and majority
vote in all settings except the IMO Shortlist, where characteristics of our verifier set imply that a naive ensemble is effectively an oracle
ensemble (see Section 3.3 for details).
and ensure that the problem is identified in the absence of
ground-truth labels. In practice, we use certain heuristics—
described in Appendix D—to drop verifiers that do not ap-
pear to be better than random so as to better ensure Assump-
tion 2.1 is satisfied. Conditional independence assumptions
like Assumption 2.2 are also common in these literatures
(Dawid & Skene, 1979; Parisi et al., 2014; Tenzer et al.,
2022; Jaffe et al., 2015), and also ensure identification. Un-
der these two assumptions, Jaffe et al. (2015) show that
verifier sensitivities and specificities can be extracted from
a rank-one structure that arises in certain covariance tensors,
as summarized by the following:
Theorem 2.3 (Jaffe et al. (2015, Lemmas 1, 2, and 4)). Let
µ denote the vector of mean values of verifier predictions for
q and Σ and T denote the second and third order marginal
covariance tensors between verifier outputs for query q; i.e.,
µ ∈Rm, Σ ∈Rm×m, T ∈Rm×m×m with entries given
by µj1 := E[vj1(q, r)],
Σj1,j2 = E
" 2
Y
ℓ=1
(vjℓ(q, r) −E[vjℓ(q, r)])
#
,
T j1,j2,j3 = E
" 3
Y
ℓ=1
(vjℓ(q, r) −E[vjℓ(q, r)])
#
,
for all j1, j2, j3 ∈[m]. Letting b := P(y(q, r) = 1) −
P(y(q, r) = −1) denote the class imbalance for q, under
Assumptions 2.1 and 2.2:
(i) The off-diagonal entries of Σ are equal to those of the
outer product matrix uu⊤, where
u :=
p
1 −b2(2π −1).
(1)
Consequently, u is identifiable from Σ up to a sign,
which is uniquely determined by Assumption 2.1.
(ii) The off-diagonal entries of T (i.e., those with all three
indices distinct) are equal to those of the rank-one third
order tensor w ⊗w ⊗w where
w := (−2b(1 −b2))1/3(2π −1).
(2)
(iii) The sensitivities and specificities of verifier outputs for
responses to query q can be written as:
ψ = 1
2
 
1 + µ + u
r
1 −b
1 + b
!
,
η = 1
2
 
1 −µ + u
r
1 −b
1 + b
!
.
(3)
Theorem 2.3 shows how the sensitivities ψ and specificities
η are uniquely determined by the marginal moment tensors
µ, Σ, T . In particular, parts (i) and (ii) of Theorem 2.3
show how to recover the vectors u and w in equations (1)
and (2) from the second and third order covariance tensors.
Because these vectors differ only by a scale factor depending
on b, this in turn recovers the class imbalance b. Finally,
per equation (3), the sensitivities and specificities can be
extracted on the basis of u, b, and the mean vector µ. Of
course, the moment tensors µ, Σ, T are unknown, but their
empirical counterparts can be computed from V and are
consistent at Op(1/
√
N) rates. Consequently, Theorem 2.3
provides a recipe for producing sensitivity and specificity
estimates ˆψ, ˆη with zero ground-truth data.
With such estimates in hand, under the additional assump-
tion that verifiers are jointly conditionally independent, Jaffe
3

## Page 4

FUSE: Ensembling Verifiers with Zero Labeled Data
Step 2
gτ*:
τ*
MoM(
)
V
̂ψ
̂η
[Jaffe et al., 2015]
fθ*:
̂S(τ)
τ
V
˜
V
̂
Acc (θ)
θ
θ*
Final label 
predictions
Minimize TCI
Maximize Est. Acc.
Step 1
Selected 
response
Step 3
{gτ : ℝN×m →{±1}N×m}
{fθ : ℝN×m →[0,1]N}
ri*
N responses
m verifiers
Figure 2. Overview of FUSE: given the matrix of verifier scores V for query q, it first finds a transformation gτ∗that minimizes an
empirical measure of TCI violation and transforms scores according to it (Step 1). It then uses the moment-based method of Jaffe
et al. (2015) to produce estimates of the query-specific sensitivities and specifities ˆψ, ˆη (Step 2). Finally, FUSE uses these estimates
to construct an estimated accuracy of any predictor fθ : RN×m →[0, 1]N, optimizing this metric across the parametric family {fθ}
ensembles to obtain a final ensemble fθ⋆and returning the response with highest predicted correctness under fθ⋆(Step 3).
et al. (2015) obtain closed-form coefficients for a near-
optimal weighted ensemble (see Appendix B for details).
Unfortunately, this ensemble does poorly in practice. As
depicted in Figure 3, on BoN test-time scaling data from
Saad-Falcon et al. (2025a), the Jaffe et al. (2015) approach
under-performs a simple naive ensemble in 7 out of 10
settings, suggesting that its assumptions are untenable in
realistic LLM verification settings. See Figure 4 for condi-
tional correlation plots on Saad-Falcon et al. (2025b) data
that confirm this.
2.2. Adaptive score transformations for enhanced MoM
estimation
To adapt the approach in the previous section to a more
general setting in which verifiers may be conditionally de-
pendent and emit real-valued scores, we establish:
Proposition 2.4. Given a set of verifiers, let Σ and T denote
the second and third order covariance tensors associated
with query q. Then, under the TCI condition stated in As-
sumption 2.2,
m
X
j3=3
Var
 T j1,j2,j3
Σj1,j2

1≤j1<j2<j3
!
= 0,
(4)
where Var(·) denotes the sample variance over the indexed
collection.
Proposition 2.4 provides a measure of violations of TCI that
does not require ground-truth labels: a large value of the
left-hand side of (4) certifies that TCI does not hold. In our
setting, we use this statistic to measure how well transforma-
tions of verifiers satisfy TCI. Specifically, let gj : R →R
denote any monotone transformation of the jth verifier’s
outputs. Letting g(V) := (gj(vi,j))i,j denote the matrix of
transformations of the original scores V, we use g(V) to
construct empirical estimates of the second- and third-order
population covariance tensors between the binarized ver-
ifiers g1(v1(·)), . . . , gm(vm(·)). Plugging these estimates
into (4) yields an empirical approximation of (4)—which
we denote by ˆS(g(V))—that provides a (feasible) measure
of deviations from TCI among the transformed verifiers.1
Binary transformation By default, we transform ver-
ifiers by binarizing verifier outputs to {±1} using a
vector τ of verifier-specific thresholds: gj,τj(vj(·)) =
sign (vj(·) −τj).
We then find τ ⋆which (approxi-
mately) minimizes ˆS(τ) :=
ˆS(gτ(V)) via coordinate
descent and apply the MoM procedure from the previ-
ous section on the matrix gτ(V) to estimate the sensi-
tivities and specificities of the τ ⋆-thresholded verifiers
g1,τ ⋆
1 (v1(·)), . . . , gm,τ ⋆
m(vm(·)).2 Additional examples of
transformations are given in Appendix D.
We write ˜V to denote the transformed matrix of verifier
scores (i.e., ˜V = gτ ⋆(V)). Correspondingly, ˜vj denotes
the transformed jth verifier (i.e., ˜vj(q, r) = gj,τ ⋆
j (vj(q,r ))).
Applying the MoM procedure of Section 2.1 to the matrix ˜V
then yields estimates ˆψ, ˆη of the q-conditional sensitivities
and specificities of the transformed verifiers ˜v1, . . . , ˜vm.
1In practice, we clip the elements in the denominator of (4) to
a small positive number to promote stability.
2For faster optimization, gradient descent can be applied by
approximating the sign function with a sigmoid.
4

## Page 5

FUSE: Ensembling Verifiers with Zero Labeled Data
0
4
8
12
16
+6.5
0
5
10
15
+6.3
0
10
20
30
+12.3
0
5
10
15
20
25
+2.7
0
10
20
+1.7
0
10
20
+6.4
0
10
20
+8.1
0
5
10
15
+2.3
0
4
8
12
16
+2.6
0
5
10
15
20
25
+5.1
Accuracy Improvement over Pass@1 (%)
GPQA 8B
Pass@1 = 28.3%
GPQA Diamond 8B
Pass@1 = 28.3%
MATH500 8B
Pass@1 = 49.9%
MMLU 8B
Pass@1 = 64.1%
MMLU Pro 8B
Pass@1 = 46.6%
GPQA 70B
Pass@1 = 42.9%
GPQA Diamond 70B
Pass@1 = 42.3%
MATH500 70B
Pass@1 = 78.0%
MMLU 70B
Pass@1 = 82.6%
MMLU Pro 70B
Pass@1 = 69.9%
Naive Ensemble
Jaffe et al. (2015)
FUSE
Figure 3. Accuracy of Jaffe et al. (2015) versus a naive ensemble and FUSE for response selection on data from Saad-Falcon et al.
(2025b), in which generator models are Llama 3.3 8B Instruct and Llama 3.3 70B Instruct. All bars are re-scaled to indicate improvement
over Pass@1. The black arrow and accompanying number indicates the accuracy gain of FUSE over Jaffe et al. (2015).
2.3. Ensemble construction
The final step of FUSE is to use the estimated verifier sen-
sitivities and specificities to construct an ensemble. We
will again depart from Jaffe et al. (2015) who impose the
stronger JCI assumption to construct an ensemble. Rather,
we will use the estimates ˆψ, ˆη to measure the quality of any
given ensemble and subsequently optimize this estimated
quality measure in a manner that is not tied to JCI. To illus-
trate the idea, temporarily suppose that the posterior label
probability
p⋆(r) := P(y(q, r) = 1 | ˜v1(q, r), . . . , ˜vm(q, r))
(5)
for query q were known. Now, consider any family of
predictors fθ indexed by parameters θ. Concretely, in our
experiments we take fθ to be the class of logistic regression
classifiers operating on the design matrix V. Which choice
of parameters θ leads to the most accurate predictions of the
ground-truth labels? We propose to measure the quality of
any given θ using the following objective
N
X
i=1
(2p⋆(ri) −1) ˆfθ(Vi•),
(6)
where ˆfθ(Vi•) is the {±1}-valued prediction of fθ given
the (original) verifier outputs for the ith response (i.e., the
ith row of V). Intuitively, parameter values which achieve
larger values of this objective correspond to better ensem-
bles. Of course, the objective in (6) is infeasible to com-
pute since the probabilities p⋆(ri) are unknown. Therefore
FUSE substitutes these oracle probabilities with estimates
ˆp(ri) derived from the estimated sensitivities/specificities
ˆψ, ˆη—we will soon describe precisely how the estimates
ˆp(ri) are computed from ˆψ, ˆη. In particular, the final en-
semble that FUSE constructs is given by fθ⋆, where
θ⋆:= arg max
θ
N
X
i=1
(2ˆp(ri) −1) ˆfθ(Vi•) =: d
Acc(θ). (7)
Once θ⋆has been computed, FUSE selects the final re-
sponse to generate as the one with the greatest likeli-
hood of being correct as measured by fθ⋆; i.e., ri⋆where
i⋆:= arg maxN
i=1 fθ⋆(Vi•).
Posterior probability estimation To construct estimates
ˆp(ri) of the posterior probabilities, we first estimate the
posterior probabilities of the correctness label given any
three verifiers. A direct calculation shows that for any three
verifiers ˜vj1, ˜vj2, ˜vj3, the posterior label probability
P(y(q, r) = 1 | ˜vj1(q, r), ˜vj2(q, r), ˜vj3(q, r))
(8)
can be expressed, under TCI, in terms of the sensitivities
ψj1, ψj2, ψj3, specificities ηj1, ηj2, ηj3, and class imbalance
b. The exact form of this relationship is given in Propo-
sition C.1 in Appendix C. Plugging in estimates of label
imbalance ˆb as well as sensitivity and specificity estimates
ˆψ, ˆη then results in an estimate of this posterior label prob-
ability: we refer to this estimate as ˆpj1,j2,j3(r). Unfortu-
nately, ˆpj1,j2,j3(r) may in general be a poor estimate of the
full posterior label probability p⋆(r). Our default remedy
(see Appendix E for alternatives) is to average posterior
estimates across triplets. That is, we estimate p⋆(r) as:
ˆp(r) =
1
 m
3

X
1≤j1<j2<j3≤m
ˆpj1,j2,j3(r).
(9)
5

## Page 6

FUSE: Ensembling Verifiers with Zero Labeled Data
See Figure 2 for a high-level visualization of FUSE, and
Algorithm 1 for pseudocode. Finally, for an extension to
the ‘batched’ case in which scores for multiple queries are
jointly used to select responses, see Appendix D.
Algorithm 1 Fully Unsupervised Score Ensembling
1: Input: Score matrix V ∈RN×m of verifier scores for
responses (ri)N
i=1 to query q; user-defined threshold
family {gτ : RN×m →{±1}N×m}; user-specified
ensemble family {fθ : RN×m →[0, 1]N}.
2: Compute τ ⋆∈arg minτ∈T ˆS(gτ(V)).
3: Set ˜V ←gτ ⋆(V) ∈{±1}N×m.
4: Apply Theorem 2.3 to ˜V to obtain (ˆψ, ˆη,ˆb).
5: Define the estimated ensemble accuracy d
Acc(θ) in (7)
using (9).
6: Compute θ⋆∈arg maxθ∈Θ d
Acc(θ).
7: Return ri⋆where i⋆←arg maxi∈[N] fθ⋆(Vi•).
3. Results
In this section, we compare the performance of FUSE to
that of semi-supervised and unsupervised baselines in the
context of BoN test-time scaling. Our basic setup, detailed
further in Appendix E, is as follows:
Baselines We consider three semi-supervised baselines,
which are each given access to ground truth labels for 5%
of queries. These are logistic regression, naive Bayes, and
WEAVER (Saad-Falcon et al., 2025a), which uses ideas
from the weak supervision literature to estimate ensem-
ble weights. Unsupervised baselines include majority vote,
which selects the most common answer among the repeated
generations, and naive ensemble, which selects the response
with the highest average verifier score. Finally, we compute
several oracle baselines such as Pass@k and the best verifier
by ground-truth accuracy. Detailed baseline descriptions
are in Appendix E.1. In Appendix E.7, we mathematically
justify omitting repeated sampling from a single verifier,
which is an obvious but non-competitive baseline. Finally,
Appendix E.8 compares FUSE to additional unsupervised
baselines such as Dawid & Skene (1979).
Models and datasets We consider three data and model
sources.
• Data from Saad-Falcon et al. (2025a) We run our
method on the exact data used in Saad-Falcon et al.
(2025a), consisting of 100 generations per question
by Llama 3.1 8B Instruct and Llama 3.3 70B Instruct
on GPQA, GPQA Diamond, MATH500, MMLU, and
MMLU Pro, with verifications by up to 33 open-source
reward models and language models.
• Humanity’s Last Exam A 649-question subsample
of Humanity’s Last Exam (Phan et al., 2025). See
Appendix E.4 for details on construction. We sample
50 generations per question from Gemini 3 Pro Pre-
view, and verifications from seven closed-source and
open-source models (see Appendix E.4).
• IMO Shortlist A 123-question subset of IMO An-
swerBench (Luong et al., 2025) comprised of modified
IMO Shortlist questions. We sample 50 generations
per question from the open-source model Qwen3-30B-
A3B-Thinking-2507, and verifications from 9 open-
source language models.
An attractive feature is that these settings exhibit vast dif-
ferences in task difficulty and in the strength of generator
and verifier models. Consequently, our method’s strong
overall performance provides evidence for usefulness across
problem ranges. In an additional suite of ablations, we
verify that the ability of our method to operate on a prompt-
conditional basis improves performance versus baselines in
‘mixed’ settings with task heterogeneity.
3.1. 70B and 8B experiments
Table 1 summarizes the Best-of-100 performance of FUSE
and baselines on data from Saad-Falcon et al. (2025b).
FUSE is competitive with semi-supervised baselines In
the 70B setting, we see that FUSE achieves essentially exact
parity with WEAVER, with minor deviations in performance
across benchmarks (for instance, 64.4% on GPQA Diamond
for FUSE vs 64.1% for WEAVER). Overall, across 70B and
8B settings, FUSE wins 27 out of 40 comparisons against
supervised baselines, and always outperforms the natural
unsupervised baseline of naive ensemble and majority vote.
Diverse verification is necessary for strong performance
Across all benchmarks, the verification-free baseline of ma-
jority vote is uniformly non-competitive, with gaps in perfor-
mance versus FUSE that range from 10.0% on 8B GPQA
to 17.0% on MMLU Pro. Further, FUSE outperforms the
oracle ‘best verifier’ in all but one setting, suggesting that
using diverse verifiers produces meaningful benefits.
Prompt-level conditioning bolsters performance in set-
tings with high heterogeneity We conduct a mixed-data
ablation to simulate the effects of high task heterogeneity
and potential distribution shift; see Table 2. Our dataset con-
sists of the first 100 questions from each of GPQA, GPQA
Diamond, MATH500, MMLU, and MMLU Pro. In the
‘Mixed Labels’ setting, semi-supervised methods are given
labels from the first 5 questions in each benchmark (5%
total). In the ‘GPQA-Only’ setting, labels are derived ex-
clusively from GPQA. We see that FUSE outperforms all
baselines in all settings, and that in the 8B setting, transi-
tioning from ‘Mixed Labels’ to ‘GPQA-Only’ widens the
6

## Page 7

FUSE: Ensembling Verifiers with Zero Labeled Data
Table 1. Best-of-100 accuracies for all baselines on data from Saad-Falcon et al. (2025b), wherein generator models are Llama 3.1 8B
Instruct and Llama 3.3 70B Instruct. Supervised methods are given access to ground-truth labels for all 100 responses to 5% of questions
(e.g. 2500 labels on MATH500). The OBV (Oracle Best Verifier) column corresponds to the selection accuracy of the verifier with highest
balanced accuracy, which may vary from benchmark to benchmark.
Unsupervised
Supervised
Oracle
Benchmark (70B)
Pass@1
Majority Vote
Naive Ensemble
FUSE
OBV
Weaver
Logistic
Naive Bayes
Pass@100
GPQA
42.9
47.4
60.1
66.8
59.1
66.4
69.3
60.5
81.0
GPQA Diamond
42.3
49.5
57.6
64.4
50.8
64.1
63.3
57.1
75.3
MATH500
78.0
82.4
92.4
92.8
87.2
93.4
96.2
94.4
98.6
MMLU
82.6
84.1
92.4
94.1
88.4
94.9
93.1
93.7
96.0
MMLU Pro
69.9
74.4
87.0
91.4
85.6
90.2
91.8
91.4
92.0
Unsupervised
Supervised
Oracle
Benchmark (8B)
Pass@1
Majority Vote
Naive Ensemble
FUSE
OBV
Weaver
Logistic
Naive Bayes
Pass@100
GPQA
28.3
30.5
37.8
40.5
41.9
47.1
35.9
37.6
95.2
GPQA Diamond
28.3
32.3
38.4
42.7
40.6
46.5
34.3
37.5
95.0
MATH500
49.9
69.6
75.2
83.9
82.8
74.8
88.0
80.8
99.2
MMLU
64.1
72.7
81.9
84.9
83.6
85.7
80.4
77.8
98.5
MMLU Pro
46.6
56.4
67.2
69.8
66.5
67.2
65.2
66.0
96.8
Table 2. Performance of methods in a mixed-data setting with 100
questions each from GPQA, GPQA Diamond, MATH500, MMLU,
and MMLU Pro. In the mixed labels setting, 5 questions from each
benchmark are held out as a labeled train set. In the GPQA-only
setting, 25 questions from GPQA are held out.
Setting
Method
8B acc. 70B acc.
Mixed Labels
Pass@1
40.2%
64.0%
Pass@100
97.1%
87.6%
Majority Vote
52.2%
67.1%
Naive Ensemble 63.0%
79.4%
WEAVER
60.1%
78.3%
FUSE
64.4% 81.8%
GPQA-Only Labels Pass@1
41.1%
65.3%
Pass@100
97.1%
88.8%
Majority Vote
53.5%
68.0%
Naive Ensemble 64.2%
80.8%
WEAVER
58.4%
79.6%
FUSE
65.2% 82.4%
gap between FUSE and WEAVER from 4.3% to 6.8%.
3.2. Humanity’s Last Exam
FUSE performs well in extremely hard settings The ex-
perimental setup in Saad-Falcon et al. (2025a) consists of
benchmarks which, while standard, are not commensurate
with the reasoning quality of recent language models. We
assess FUSE and baselines on a curated 649-question sub-
set of Humanity’s Last Exam (Phan et al., 2025), which
is presently unsaturated by frontier closed-source models
such as Gemini 3 Pro and GPT 5.2 Pro. In constructing this
subset, we exclude questions with zero correct responses so
as to make comparisons between selection methods mean-
ingful.
Table 3. Best-of-50 performance of methods on Humanity’s Last
Exam.
Method
Accuracy
Pass@1
52.1%
Naive Ensemble
51.4%
Oracle Best Verifier (GPT-5.2 High)
53.5%
Naive Bayes
52.0%
Logistic Regression
53.4%
WEAVER
51.2%
FUSE
54.3%
In Table 3, only the oracle best verifier, semi-supervised
logistic regression, and FUSE outperform the Pass@1 base-
line. Notably, naive ensemble is worse than simply selecting
a random response, validating the extreme difficulty of this
setting.
3.3. IMO Shortlist
In our final main setting, we consider the IMO Shortlist
subset of IMO AnswerBench (Luong et al., 2025), which
consists of past IMO Shortlist questions that are expert-
modified to prevent memorization from training data. This
setting, in contrast to the previous ones, features substan-
tial homogeneity in verifier strength and near-conditional
independence of verifiers (see Appendix E.5). As such, the
7

## Page 8

FUSE: Ensembling Verifiers with Zero Labeled Data
Table 4. Best-of-50 performance of methods on IMO Shortlist
Method
Accuracy
Pass@1
53.3%
Majority Vote
57.7%
Naive Ensemble
63.8%
Oracle Best Verifier (gpt-oss-20b)
59.7%
Naive Bayes
59.1%
Logistic Regression
60.2%
WEAVER
62.1%
FUSE
63.8%
naive ensemble baseline is effectively an oracle baseline.
We see in Table 4 that FUSE is the only method to match
a naive ensemble, while other data-dependent methods fall
behind. This setting, besides having highly similar verifiers,
also has substantially fewer verifiers than the Saad-Falcon
et al. (2025b) experiments (9 vs 33). Therefore, neither a
high verifier count nor extreme verifier heterogeneity are
required for our method to perform well.
4. Related work
4.1. Verification for test-time scaling
A large number of works have studied variants of BoN
for improved test-time scaling when using a single verifier
model (e.g., Cobbe et al., 2021; Nakano et al., 2021; Ichi-
hara et al., 2025; Jinnai et al., 2024; Rakhsha et al., 2025).
Like these works, FUSE aims to achieve response accu-
racy as close as possible to the Pass@k rate (Chen, 2021),
but differs in that it uses multiple verifier models to boost
accuracy.
Prior work on leveraging multiple verifiers to improve veri-
fication quality includes Verga et al. (2024); Lifshitz et al.
(2025); Saad-Falcon et al. (2025a), who propose various
unweighted and semi-supervised rules for score aggregation.
Such rules are either too simple to account for differing
abilities and statistical dependencies between verifiers or
rely on holdout sets of labeled data. In contrast, FUSE
constructs effective and data-adaptive ensembles with zero
access to ground truth labels. Lastly, we remark that in a
distinct but related setting, works such as Coste et al. (2023);
Eisenstein et al. (2023) study how ensembling verifiers can
reduce reward-hacking in reinforcement learning.
4.2. Unsupervised ensembling
As detailed in Section 2, the core methodological ideas un-
derlying FUSE stem from the literature on ensembling ML
models with no access to labeled data (Parisi et al., 2014;
Jaffe et al., 2015; 2016; Tenzer et al., 2022). These works all
require conditional independence assumptions which empir-
ically fail to hold in the LLM verification setting, whereas
we adaptively manage such violations so that a Jaffe et al.
(2015)-inspired estimation procedure can be applied.
5. Discussion
We introduced FUSE, a method for unsupervised ensem-
bling of verifiers in settings where one has access to repeated
samples from a generator model. Our experiments involve
settings which range from conventional (e.g. MMLU) to
frontier (e.g. Humanity’s Last Exam) in difficulty. Regard-
less of setting, FUSE competes well with baselines that
require ground-truth labels. An attractive feature of our
method, which also boosts performance when tasks are di-
verse, is that FUSE supports both query-conditional and
batched modes of operation.
The primary limitation of our method is its computational
and logistical efficiency. As presented, our method is cum-
bersome in that it demands sampling from distinct verifier
models. In Saad-Falcon et al. (2025a), distillation of ensem-
ble predictions into a small model is proposed as way to
reap the benefits of diverse verification in a FLOP-efficient
manner. A similar solution could be implemented in our
setting, but any fixed model would be susceptible to dis-
tribution shift, whereas the question-conditional version of
FUSE is immune by design.
We expect our ideas to have applicability beyond the test-
time scaling setting, and record some promising directions:
• Unsupervised model ranking Robust scoring of re-
sponses without ground-truth labels may enable un-
supervised model ranking and exciting new forms of
benchmarks. For instance, Nie et al. (2025) propose
benchmarking language models through LLM-based
scoring of attempts at unsolved questions. The credi-
bility of this scheme is unavoidably tied to the strength
of available scoring protocols, which FUSE and future
refinements would boost.
• Benchmark and ground-truth auditing Many bench-
marks have label errors which require human experts
to identify, for which FUSE could provide diagnostics
by isolating inconsistencies between answer key cor-
rectness and estimated correctness. Similarly, FUSE
could be used to rank sources of ground-truth such as
expert human labelers on challenging tasks.
• Data filtering Unsupervised, high-quality scoring of
model generations is of potential relevance in any task
where data filtering is necessary, such as selection of
synthetic training data and reinforcement learning with
rubric-based rewards.
For many tasks, model capability and the cost of useful
8

## Page 9

FUSE: Ensembling Verifiers with Zero Labeled Data
ground-truth labels will inevitably rise in tandem. The most
ambitious conceptual response to this marriage of cost and
utility is dispensing with supervision altogether. We do not
expect this to be possible in all circumstances and settings,
but view FUSE as providing initial validation that the full
potential of unsupervised methods lies beyond the present
frontier.
6. Acknowledgments
We thank Andrew Ilyas, Sarah Cen, and Zitong Yang for
helpful discussions. This work was partially conducted
using the Stanford Marlowe GPU cluster (Kapfer et al.,
2025). Y.N. and A.S. were both supported by a National
Science Foundation Graduate Research Fellowship. A.S.
was also supported by the Citadel GQS PhD fund. V.M.
was supported by the Stanford Data Science Scholarship.
E.J.C. was supported by the Office of Naval Research (grant
N00014-24-1-2305) and the National Institutes of Health
(grant 1R01AG08950901A1). We gratefully acknowledge
the support of Google and Google Cloud.
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.
References
Ahsen, M. E., Vogel, R. M., and Stolovitzky, G. A. Unsu-
pervised evaluation and weighted aggregation of ranked
classification predictions. Journal of Machine Learning
Research, 20(166):1–40, 2019.
Box, G. E. and Cox, D. R. An analysis of transformations.
Journal of the Royal Statistical Society Series B: Statisti-
cal Methodology, 26(2):211–243, 1964.
Chen, M. Evaluating large language models trained on code.
arXiv preprint arXiv:2107.03374, 2021.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
R., et al. Training verifiers to solve math word problems.
arXiv preprint arXiv:2110.14168, 2021.
Coste, T., Anwar, U., Kirk, R., and Krueger, D. Reward
model ensembles help mitigate overoptimization. arXiv
preprint arXiv:2310.02743, 2023.
Dawid, A. P. and Skene, A. M. Maximum likelihood es-
timation of observer error-rates using the em algorithm.
Journal of the Royal Statistical Society: Series C (Applied
Statistics), 28(1):20–28, 1979.
Di, Q., Ji, K., Li, X., Zhao, H., and Gu, Q. Best-of-majority:
Minimax-optimal strategy for pass@ k inference scaling.
arXiv preprint arXiv:2510.03199, 2025.
Didwania, Y., Nair, J., and Hemachandra, N. Unsupervised
crowdsourcing with accuracy and cost guarantees. In
2022 20th International Symposium on Modeling and
Optimization in Mobile, Ad hoc, and Wireless Networks
(WiOpt), pp. 137–144. IEEE, 2022.
Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A.,
D’Amour, A., Dvijotham, D., Fisch, A., Heller, K., Pfohl,
S., Ramachandran, D., et al. Helping or herding? reward
model ensembles mitigate but do not eliminate reward
hacking. arXiv preprint arXiv:2312.09244, 2023.
Gunjal, A., Wang, A., Lau, E., Nath, V., He, Y., Liu, B.,
and Hendryx, S. Rubrics as rewards: Reinforcement
learning beyond verifiable domains, 2025. URL https:
//arxiv.org/abs/2507.17746.
Ichihara, Y., Jinnai, Y., Morimura, T., Ariu, K., Abe, K.,
Sakamoto, M., and Uchibe, E. Evaluation of best-of-n
sampling strategies for language model alignment. arXiv
preprint arXiv:2502.12668, 2025.
Jaffe, A., Nadler, B., and Kluger, Y. Estimating the ac-
curacies of multiple classifiers without labeled data. In
Lebanon, G. and Vishwanathan, S. V. N. (eds.), Pro-
ceedings of the Eighteenth International Conference on
Artificial Intelligence and Statistics, volume 38 of Pro-
ceedings of Machine Learning Research, pp. 407–415,
San Diego, California, USA, 09–12 May 2015. PMLR.
URL https://proceedings.mlr.press/v38/
jaffe15.html.
Jaffe, A., Fetaya, E., Nadler, B., Jiang, T., and Kluger, Y. Un-
supervised ensemble learning with dependent classifiers.
In Gretton, A. and Robert, C. C. (eds.), Proceedings of the
19th International Conference on Artificial Intelligence
and Statistics, volume 51 of Proceedings of Machine
Learning Research, pp. 351–360, Cadiz, Spain, 09–11
May 2016. PMLR. URL https://proceedings.
mlr.press/v51/jaffe16.html.
Jinnai, Y., Morimura, T., Ariu, K., and Abe, K. Regular-
ized best-of-n sampling to mitigate reward hacking for
language model alignment. In ICML 2024 Workshop on
Models of Human Feedback for AI Alignment, 2024.
Kapfer, C., Stine, K., Narasimhan, B., Mentzel, C., and
Candes, E. Marlowe: Stanford’s GPU-based compu-
tational instrument.
Zenodo, 2025.
URL https:
//doi.org/10.5281/zenodo.14751899.
Kleindessner, M. and Awasthi, P. Crowdsourcing with ar-
bitrary adversaries. In International Conference on Ma-
chine Learning, pp. 2708–2717. PMLR, 2018.
9

## Page 10

FUSE: Ensembling Verifiers with Zero Labeled Data
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu,
C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient
memory management for large language model serving
with pagedattention. In Proceedings of the ACM SIGOPS
29th Symposium on Operating Systems Principles, 2023.
Lifshitz, S., McIlraith, S. A., and Du, Y. Multi-agent verifi-
cation: Scaling test-time compute with multiple verifiers.
arXiv preprint arXiv:2502.20379, 2025.
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker,
B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and
Cobbe, K. Let’s verify step by step. In The Twelfth
International Conference on Learning Representations,
2023.
Liu, T., Zhao, Y., Joshi, R., Khalman, M., Saleh, M.,
Liu, P. J., and Liu, J.
Statistical rejection sampling
improves preference optimization. In The Twelfth In-
ternational Conference on Learning Representations,
2024. URL https://openreview.net/forum?
id=xbjSwwrQOe.
Luong, T., Hwang, D., Nguyen, H. H., Ghiasi, G., Cher-
vonyi, Y., Seo, I., Kim, J., Bingham, G., Lee, J., Mishra,
S., Zhai, A., Hu, H., Michalewski, H., Kim, J., Ahn,
J., Bae, J., Song, X., Trinh, T. H., Le, Q. V., and
Jung, J. Towards robust mathematical reasoning. In
Christodoulopoulos, C., Chakraborty, T., Rose, C., and
Peng, V. (eds.), Proceedings of the 2025 Conference
on Empirical Methods in Natural Language Process-
ing, pp. 35418–35442, Suzhou, China, November 2025.
Association for Computational Linguistics. ISBN 979-
8-89176-332-6.
doi: 10.18653/v1/2025.emnlp-main.
1794. URL https://aclanthology.org/2025.
emnlp-main.1794/.
Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim,
C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., et al.
Webgpt: Browser-assisted question-answering with hu-
man feedback. arXiv preprint arXiv:2112.09332, 2021.
Nie, F., Liu, K. Z., Wang, Z., Sun, R., Liu, W., Shi, W., Yao,
H., Zhang, L., Ng, A. Y., Zou, J., Koyejo, S., Choi, Y.,
Liang, P., and Muennighoff, N. Uq: Assessing language
models on unsolved questions, 2025. URL https://
arxiv.org/abs/2508.17580.
Parisi, F., Strino, F., Nadler, B., and Kluger, Y.
Rank-
ing and combining multiple predictors without labeled
data.
Proceedings of the National Academy of Sci-
ences, 111(4):1253–1258, 2014.
doi: 10.1073/pnas.
1219097111. URL https://www.pnas.org/doi/
abs/10.1073/pnas.1219097111.
Phan, L., Gatti, A., Han, Z., Li, N., et al. Humanity’s
last exam, 2025. URL https://arxiv.org/abs/
2501.14249.
Rakhsha, A., Madan, K., Zhang, T., Farahmand, A.-m., and
Khasahmadi, A. Majority of the bests: Improving best-
of-n via bootstrapping. arXiv preprint arXiv:2511.18630,
2025.
Saad-Falcon, J., Buchanan, E. K., Chen, M. F., Huang, T.-
H., McLaughlin, B., Bhathal, T., Zhu, S., Athiwaratkun,
B., Sala, F., Linderman, S., Mirhoseini, A., and R´e,
C. Shrinking the generation-verification gap with weak
verifiers, 2025a. URL https://arxiv.org/abs/
2506.18203.
Saad-Falcon, J., Buchanan, E. K., Chen, M. F., Huang, T.-H.,
McLaughlin, B., Bhathal, T., Zhu, S., Athiwaratkun, B.,
Sala, F., Linderman, S., et al. Shrinking the generation-
verification gap with weak verifiers.
arXiv preprint
arXiv:2506.18203, 2025b.
Shaham, U., Cheng, X., Dror, O., Jaffe, A., Nadler, B.,
Chang, J., and Kluger, Y. A deep learning approach to
unsupervised ensemble learning. In International confer-
ence on machine learning, pp. 30–39. PMLR, 2016.
Steinhardt, J. and Liang, P. S. Unsupervised risk estimation
using only conditional independence structure.
In
Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., and
Garnett, R. (eds.), Advances in Neural Information
Processing Systems, volume 29. Curran Associates, Inc.,
2016.
URL https://proceedings.neurips.
cc/paper_files/paper/2016/file/
f2d887e01a80e813d9080038decbbabb-Paper.
pdf.
Sun, H., Haider, M., Zhang, R., Yang, H., Qiu, J., Yin, M.,
Wang, M., Bartlett, P., and Zanette, A. Fast best-of-n
decoding via speculative rejection. Advances in Neural
Information Processing Systems, 37:32630–32652, 2024.
Tenzer,
Y.,
Dror,
O.,
Nadler,
B.,
Bilal,
E.,
and
Kluger, Y.
Crowdsourcing regression:
A spectral
approach.
In Camps-Valls, G., Ruiz, F. J. R., and
Valera, I. (eds.), Proceedings of The 25th Interna-
tional Conference on Artificial Intelligence and Statis-
tics, volume 151 of Proceedings of Machine Learn-
ing Research, pp. 5225–5242. PMLR, 28–30 Mar
2022. URL https://proceedings.mlr.press/
v151/tenzer22a.html.
Verga, P., Hofstatter, S., Althammer, S., Su, Y., Piktus,
A., Arkhangorodsky, A., Xu, M., White, N., and Lewis,
P. Replacing judges with juries: Evaluating llm gener-
ations with a panel of diverse models. arXiv preprint
arXiv:2404.18796, 2024.
Zhao, E., Awasthi, P., and Gollapudi, S. Sample, scrutinize
and scale: Effective inference-time search by scaling
verification. arXiv preprint arXiv:2502.01839, 2025.
10

## Page 11

FUSE: Ensembling Verifiers with Zero Labeled Data
A. Further details on MoM estimation of sensitivities and specificities
In this section, we elaborate on further details of the MoM estimation procedure of Jaffe et al. (2015). Section A.1 formalizes
the TCI condition in Assumption 2.2 and Section A.2 explains how the procedure and surrounding results can be extended
to real-valued scores.
A.1. Triplet conditional independence
Formally, the TCI condition stated in Assumption 2.2 is that, for each triplet of distinct indices j1, j2, j3 ∈[m] and every
aj1, aj2, aj3, y ∈{±1}, we have
P(vj1(q, r) = aj1, vj2(qk, r) = aj2, vj3(qk, r) = aj3 | y(q, r) = y) = P(vj1(q, r) = aj1 | y(q, r) = y)
× P(vj2(q, r) = aj2 | y(q, r) = y)
× P(vj3(q, r) = aj3 | y(q, r) = y).
(10)
A.2. MoM estimates for real-valued score matrices
As discussed in Section 2.1, the MoM estimator of Jaffe et al. (2015) can be extended to real-valued scores. We operate
under the assumption that all verifier scores lie in the interval [−1, 1].3 Furthermore, we extend the definition of sensitivity
and specificity in this setting to be:
ψj := E
1 + vj(q, r)
2
| y(q, r) = 1

, ηj := E
1 −vj(q, r)
2
| y(q, r) = −1

.
As before, the balanced accuracy is given by πj := ψj+ηj
2
. Notice that in the original {±1}-valued case, these definitions
coincide with those presented in the main manuscript. Finally, we require the same TCI condition as in Assumption 2.2: the
(real-valued) scores output by any three distinct verifiers are conditionally independent given the true label. We claim that,
under these definitions and conditions, Theorem 2.3—which restates Jaffe et al. (2015)’s identities relating ψ and η to the
first three moment/covariance tensors in the binary classification setting—continues to hold in this real-valued setup. The
reason is that the proofs in Parisi et al. (2014); Jaffe et al. (2015) use only the fact that ψj and ηj are equal to expectations of
(affine transformations of) verifier outputs. This relationship is entirely unchanged in our setting, meaning that their proofs
continue to go through in our real-valued setting without modification. To illustrate the idea, we provide a proof of the
extension of part (i) of Theorem 2.3.
Proof of real-valued extension of Theorem 2.3 (i). It suffices to verify that, for any j1 ̸= j2,
E[(vj1(q, r)vj2(q, r)] −E[vj1(q, r)]E[vj2(q, r)] = (1 −b2)(2πj1 −1)(2πj2 −1).
(11)
Under TCI, the left-hand side equals
1 + b
2
· (2ψj1 −1)(2ψj2 −1) + 1 −b
2
· (2ηj1 −1)(2ηj2 −1)
−
1 + b
2
· (2ψj1 −1) −1 −b
2
· (2ηj1 −1)
 1 + b
2
· (2ψj2 −1) −1 −b
2
· (2ηj2 −1)

A direct calculation shows that this is the same as the right-hand side of (11).
Finally, we note that Proposition 2.4 continues to hold in our real-valued setting, as it is directly implied by (the real-valued
extension of) Theorem 2.3.
3Accordingly, we re-scale verifier scores to lie in [−1, 1] before applying FUSE.
11

## Page 12

FUSE: Ensembling Verifiers with Zero Labeled Data
B. Ensembling under joint conditional independence (Jaffe et al., 2015)
Suppose that the true sensitivities and specificities ψ, η are known. Then—again, translating Jaffe et al. (2015)’s results and
setup to our repeated verification setting—given a response r to query q, Jaffe et al. (2015) propose a procedure to estimate
the unknown label y(q, r). In particular, under the JCI assumption that the verifier scores vj(q, r) are jointly conditionally
independent given the true label y(q, r), they show that the maximum likelihood estimate (MLE) of y(q, r) is given by
ˆy := sign


m
X
j=1
vj(q, r) log
ψj(1 −ψj)
ηj(1 −ηj)

+ log
ψj(1 −ψj)
ηj(1 −ηj)

,
(12)
As ψ and η are unknown in practice, Jaffe et al. (2015) propose to simply plug in the estimates ˆψ, ˆη into (12) to obtain an
approximation of the MLE.
C. Estimation of posterior probabilities
In this section, we show how to estimate the posterior label probabilities given any three verifier predictions.
Proposition C.1. For any three indices j1, j2, j3 ∈[m], we have that
P(y(q, ri) = y | vi,j1, vi,j2, vi,j3) ∝(1 + by)
3
Y
ℓ=1
[1 −yvjℓ+ vjℓ((1 + y)ψjℓ−(1 −y)ηjℓ)]
(13)
under the TCI condition (Assumption 2.2). Consequently, plugging in the estimates of class imbalance, sensitivities, and
specificities above leads to a consistent estimate of the posterior probability P(y(q, ri) = y | vi,j1, vi,j2, vi,j3).
Proof. The proof follows by a direct calculation:
P(y(q, ri) = 1 | vi,j1, vi,j2, vi,j3) ∝P(vi,j1, vi,j2, vi,j3 | y(q, ri) = 1)P(y(q, ri) = 1)
= P(y(q, ri) = 1)
3
Y
ℓ=1
P(vjℓ(q, ri) | y(q, ri) = 1)
= P(y(q, ri) = 1)
3
Y
ℓ=1
1 + vjℓ
2
ψjℓ+ 1 −vjℓ
2
(1 −ψjℓ)

=
1 + b
2

3
Y
ℓ=1

vjℓψjℓ+ 1 −vjℓ
2

,
where the first equality holds by TCI. Similarly,
P(y(q, ri) = −1 | vi,j1, vi,j2, vi,j3) ∝
1 −b
2

3
Y
ℓ=1
1 + vjℓ
2
(1 −ηjℓ) + 1 −vjℓ
2
ηjℓ

=
1 −b
2

3
Y
ℓ=1
1 + vjℓ
2
−vjℓηjℓ

.
In summary, we have that
P(y(q, ri) = y | vi,j1, vi,j2, vi,j3) ∝(1 + by)
3
Y
ℓ=1
[1 −yvjℓ+ vjℓ((1 + y)ψjℓ−(1 −y)ηjℓ)] ,
as was to be shown.
12

## Page 13

FUSE: Ensembling Verifiers with Zero Labeled Data
D. Additional details for FUSE
In this section, we record additional implementation details and design considerations.
Batching As mentioned in the main text, our method can operate in both query-conditional and batched modes. When
batching, we vertically concatenate score matrices V1, . . . , Vℓcorresponding to queries q1, . . . , qℓto form a ‘tall’ score
matrix V and apply FUSE to V to learn an ensemble. Then, for each query q, we return the response with the highest
predicted probability of correctness in the corresponding sub-matrix of V.
Dropping Various heuristics can be used to drop potentially poor verifiers. By default, we use a balanced-accuracy criterion.
After obtaining the transformed matrix V, we apply the method-of-moments estimate in Jaffe et al. (2015) to the entire
matrix, and drop verifiers with estimated balanced accuracy less than 1
2 before proceeding with posterior estimation. While
informally motivated, we find this to be robustly performance-enhancing. Notably, even prior works that focus on theoretical
guarantees (e.g., Tenzer et al. (2022)) use similar dropping heuristics in real-data settings.
Posterior aggregation An alternative approach to aggregating posterior estimates from triplets is to merge verifiers prior
to transformation. Intuitively, if a ‘verifier’ is the average of scores from several other verifiers, a triplet including it will
incorporate information from more than three verifiers. In the extreme case, one can imagine condensing all verifiers into a
single triplet. This approach was inspired by Steinhardt & Liang (2016), who assume in the context of unsupervised risk
estimation that a variable set can be partitioned into three ‘views’. In practice, we find simple averaging to be superior unless
the number of verifiers is large.
Cross-fitting In principle, one can form the expected accuracy objective (7) through cross-fitting—splitting the N responses
into random folds, and ensuring that posterior estimates and predictors see disjoint data. Because our setting is transductive—
we are not interested in generalization, and exclusively care about predicting an in-sample label—the traditional statistical
intuition that one must avoid label leakage has less force here. In practice, for the range of N considered in this work, we do
not find that cross-fitting produces reliable improvements in selection accuracy, and hence choose to omit it by default.
Predictors In principle, once pseudo-labels have been acquired, a natural and parameter-free selection rule exists—selecting
the response with the highest probability of correctness according to the pseudo-labels. The primary advantage of using
pseudo-labels to fit an alternative predictor is therefore not that this makes selection possible, but that the fitting process
can (i) act as a form of regularization (ii) encode favorable inductive biases and (iii) allow one to incorporate auxiliary
information or covariates that the pseudo-labeling process does not have access to. These reasons, while not fully explored
in the present text (e.g., we do not use auxiliary covariates in any of our experiments), may nonetheless enhance the general
applicability of FUSE.
Alternative Transformations Since real-valued or rubric-valued scores may encode richer information than binary ones, it
can be desirable to apply real-valued transformations that preserve this granularity.4 One way of doing so is to use Box-Cox
transformations (Box & Cox, 1964).
E. Experimental details
Several of our experiments involve generation of responses and verifications from open-source models. All such generation
was done through vLLM 0.13.0 (Kwon et al., 2023) on a compute node with 8 NVIDIA H100 GPUs (80 GB memory each).
We use the following sampling parameters:
• temperature: For generation of repeated responses, we set the temperature to be 1.0 by default. Alternative values
were used if recommended by the HuggingFace model card. For verifications, a temperature of 0.0 was used for
replicability. Note that this does not affect notions of conditional independence like TCI, since there is randomness in
responses not accounted for by y(q, r) = ±1 even when verifier signals are deterministic.
• top p: 0.95 by default. Alternative values were used if recommended by the HuggingFace model card.
• max model len: The maximum possible value for each model.
It is also necessary to extract ground truth correctness labels for the responses that we generate. For multiple choice questions,
4The intuition that binarization is strictly harmful because it ‘loses information’ is incorrect, however, as binarization can improve the
decisiveness of a verifier signal.
13

## Page 14

FUSE: Ensembling Verifiers with Zero Labeled Data
we deterministically parse the tagged final answer. For short answer questions, we used Qwen3-Next-80B-A3B-Instruct to
compare the tagged final answer to a ground truth solution. Appendix E.3 contains prompts for this ground truth extraction
pipeline, as well as for generation and verification. We hand-audited a large fraction of responses (around 25%) to check the
automated ground truth extraction and evaluation procedure.
Tie-breaking In our test-time scaling experiments, a single response must be selected out of K candidates. However, many
selection rules (e.g. picking the response with the largest logit) can produce ties. In such cases, we record the accuracy of
the selection as the fraction of tied responses which are correct.
E.1. Baselines
All experimental settings implement and report the following baselines.
• Pass@1 This baseline simply returns the first response r1 to a query. It requires neither repeated sampling nor
verification.
• Pass@k This baseline involves generating k responses r1, . . . , rk for each query, and deeming a query solved if at least
one response is correct (Chen, 2021). This baseline requires knowledge of ground truth labels and does not involve
verification.
• Majority vote Given k responses r1, . . . , rk to a query, majority vote returns the most common response r⋆. This
baseline requires neither ground truth labels nor verification.
• Naive ensemble All verifiers are weighted equally, so the score of each response is the average of its post-normalization
verifier scores. This method requires verifiers but is unsupervised. To ensure scores produced by different verifiers are
comparable, we normalize each verifier’s scores to [−1, 1] using min-max normalization.5
• WEAVER (Saad-Falcon et al., 2025b) This semi-supervised baseline uses a held-out set of queries with ground truth
labels to estimate P(yi = 1) and to adaptively binarize and drop verifiers. The ‘inner loop’ that provides parameter
estimates is gradient descent on a method-of-moments objective that assumes joint conditional independence.
• Logistic Regression This semi-supervised baseline involves fitting a logistic regression using ground truth labels for
5% of queries and verifier outputs as covariates. That is, we fit β0, β in the model P(yik = 1) = σ(βT (Vk)i• + β0)
using the defaults in scikit-learn.
• Naive Bayes This semi-supervised baseline assumes conditional independence, and selects the response with the
largest value of
P(yi = 1 | Vi•)
P(yi = −1 | Vi•) =
Q
j P(vij | yi = 1)P(yi = 1)
Q
j P(vij | yi = −1)P(yi = −1),
where all probabilities on the RHS are estimated from 5% of queries that have ground truth labels. As this baseline
assumes binary scores, we employ median binarization—the top 50% of each verifier’s scores are mapped to 1, while
the remainder are mapped to -1.
• Oracle Best Verifier The verifier with the highest balanced accuracy.
For convenience, we collect the requirements of each baseline method in Table 5.
E.2. Experiments on datasets from Saad-Falcon et al. (2025a)
We use the data available at https://huggingface.co/collections/hazyresearch/weaver with zero
modifications. Therefore, our setting consists of:
• Benchmarks GPQA, GPQA Diamond, MATH500, and subsamples of MMLU and MMLU Pro. The MMLU subsample
consists of all college-level biology, chemistry, physics, mathematics, computer science, and medicine questions. The
MMLU Pro subsample is 500 randomly chosen questions out of 12,000.
5Concretely, given scores v(q, r1), . . . , v(q, rN) for responses to a query, we map each score via x 7→2×
x−mini v(q,ri)
maxi v(q,ri)−mini v(q,ri) −1
14

## Page 15

FUSE: Ensembling Verifiers with Zero Labeled Data
Method
Requires Verifiers
Supervised
Oracle
Question-Conditional
Pass@1
No
No
No
Yes
Pass@k
No
Yes
Yes
Yes
FUSE
Yes
No
No
Yes
WEAVER (Saad-Falcon et al., 2025b)
Yes
Yes
No
No
Naive Ensemble
Yes
No
No
Yes
Majority Vote
No
No
No
Yes
Logistic Regression
Yes
Yes
No
No
Naive Bayes
Yes
Yes
No
No
Oracle Best Verifier
Yes
Yes
Yes
Yes
Table 5. Comparison of methods and requirements.
• Generator models Llama 3.1 8B Instruct and Llama 3.3 70B Instruct.
• Verifier models 33 open-source reward models and binary LM judges, which output real-valued and binary scores
respectively. When the generator model is Llama 3.1 8B Instruct, all binary LM judges are excluded. See Table 6
• Number of generations per question 100.
Running our experiment on this data solely consists of parsing it into matrices before computing the performance of
FUSE and baselines. To obtain numbers for WEAVER, we run the replication code available at https://github.
com/HazyResearch/scaling-verification6. All other baseline numbers are manually implemented, with
semi-supervised methods allowed to use labels from 5% of questions per benchmark.
The verifiers in this dataset are correlated given the true response: see Figure 4 for the average conditional correlations in
this data (conditional correlations weighted by label frequency). In particular, the correlations between score-based reward
models are quite large, showing both positive and negative correlations. This is the case both for correlations given correct
responses and given incorrect responses.
E.3. Prompts for generation, verification, and ground truth extraction
We used variants of the following two prompts (the first was adapted from Saad-Falcon et al. (2025b)) to generate responses
to a given question query.
Example generation prompt for HLE
Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}
Example generation prompt for IMO Shortlist
Your task is to answer the following question:
‘‘‘
{question}
‘‘‘
Think step by step. Enclose your final result within a pair of ‘<answer>‘ tags
without any additional formatting.
6This replication code reveals several typos in Tables 1 and 3 of Saad-Falcon et al. (2025a), which we amend in Table 1 and related
figures.
15

## Page 16

FUSE: Ensembling Verifiers with Zero Labeled Data
Qwen/Qwen2.5-72B-Instruct_verdicts
Mixtral-8x22B-Instruct-v0.1_verdicts
Meta-Llama-3.1-405B-Instruct-quantized.w8a16_verdicts
DeepSeekLlama70B_verdicts
DeepSeekQwen32B_verdicts
Llama-3.3-70B-Instruct_verdicts
SkyT1_verdicts
WizardLM-2-8x22B_verdicts
GRM_scores
URM_scores
QRM_scores
GPM_scores
GRMLlama32_scores
OffsetBias_scores
GRMGemma_scores
QwenPRM_min_scores
QwenPRM_max_scores
QwenPRM_avg_scores
InternLM2RewardModel_scores
InternLM2Reward7B_scores
Qwen72B_scores
DecisionTreeReward8B_scores
Skyworks_scores
ArmorRM_scores
EurusPRMStage1_min_scores
EurusPRMStage1_max_scores
EurusPRMStage1_avg_scores
EurusPRMStage2_min_scores
EurusPRMStage2_max_scores
EurusPRMStage2_avg_scores
QRMGemma_scores
LDLRewardGemma_scores
INFORM_scores
SkyworksGemma_scores
DecisionTreeReward27B_scores
1.0 0.1 0.3 0.1 0.1 0.2 0.0 0.1 0.1 0.1 0.1 -0.0 0.1 0.1 0.1 0.1 0.0 0.0 0.1 0.1 0.0 0.1 0.1 0.0 0.1 -0.1 -0.1 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1
0.1 1.0 0.3 0.1 0.1 0.2 0.0 0.2 0.1 0.1 0.1 -0.0 0.0 0.1 0.0 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 -0.0 -0.0 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1
0.3 0.3 1.0 0.1 0.2 0.2 0.0 0.3 0.1 0.1 0.1 -0.0 0.0 0.1 0.1 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.0 -0.0 -0.0 0.1 0.0 0.0 0.1 0.1 0.1 0.1 0.1
0.1 0.1 0.1 1.0 0.0 -0.0 0.0 0.0 0.0 0.0 0.1 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.1 0.1 0.2 0.0 1.0 0.0 0.0 0.1 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.0 -0.0 -0.0 0.0 -0.0 0.0 0.1 0.0 0.0 0.0 0.0
0.2 0.2 0.2 -0.0 0.0 1.0 -0.0 0.2 0.0 0.1 0.0 -0.1 0.0 0.1 0.0 0.0 0.0 0.1 0.1 0.1 0.0 0.1 0.0 0.0 -0.0 -0.0 -0.0 0.0 0.0 0.0 0.1 0.0 0.1 0.1 0.0
0.0 0.0 0.0 0.0 0.0 -0.0 1.0 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.1 0.2 0.3 0.0 0.1 0.2 0.0 1.0 0.1 0.0 0.1 -0.0 0.0 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.0 0.1 0.1 0.1 0.1 -0.0 -0.0 0.1 0.0 0.0 0.1 0.0 0.0 0.1 0.1
0.1 0.1 0.1 0.0 0.0 0.0 0.0 0.1 1.0 0.4 0.6 -0.1 0.5 0.6 0.5 0.1 0.1 -0.1 0.5 0.5 0.1 0.6 0.5 0.2 0.3 -0.2 -0.1 0.5 -0.0 0.4 0.5 0.4 0.5 0.5 0.5
0.1 0.1 0.1 0.0 0.0 0.1 0.0 0.0 0.4 1.0 0.5 -0.1 0.4 0.4 0.4 0.1 0.0 -0.1 0.4 0.3 0.1 0.5 0.4 0.1 0.3 -0.1 -0.1 0.4 -0.0 0.4 0.4 0.4 0.5 0.5 0.4
0.1 0.1 0.1 0.1 0.0 0.0 0.0 0.1 0.6 0.5 1.0 -0.1 0.6 0.6 0.6 0.1 0.0 -0.1 0.5 0.4 0.1 0.7 0.6 0.1 0.3 -0.2 -0.1 0.5 -0.0 0.4 0.6 0.5 0.6 0.6 0.6
-0.0 -0.0 -0.0 0.1 -0.0 -0.1 -0.0 -0.0 -0.1 -0.1 -0.1 1.0 -0.1 -0.1 -0.1 -0.0 -0.0 -0.0 -0.1 -0.1 -0.0 -0.2 -0.1 -0.1 -0.0 0.1 -0.0 -0.1 0.0 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1
0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.4 0.6 -0.1 1.0 0.5 0.6 0.1 0.0 -0.1 0.5 0.4 0.1 0.6 0.6 0.1 0.3 -0.2 -0.1 0.4 -0.1 0.4 0.5 0.5 0.6 0.6 0.5
0.1 0.1 0.1 0.0 0.0 0.1 0.0 0.1 0.6 0.4 0.6 -0.1 0.5 1.0 0.5 0.1 0.1 -0.1 0.5 0.5 0.1 0.6 0.5 0.2 0.3 -0.2 -0.1 0.5 -0.0 0.4 0.5 0.4 0.5 0.5 0.5
0.1 0.0 0.1 0.0 0.0 0.0 0.0 0.0 0.5 0.4 0.6 -0.1 0.6 0.5 1.0 0.1 0.0 -0.1 0.4 0.3 0.2 0.5 0.5 0.1 0.3 -0.2 -0.1 0.5 -0.0 0.4 0.6 0.5 0.5 0.6 0.6
0.1 0.1 0.1 0.0 0.1 0.0 0.0 0.1 0.1 0.1 0.1 -0.0 0.1 0.1 0.1 1.0 0.1 0.3 0.1 0.1 0.1 0.1 0.1 0.0 0.1 -0.0 -0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
0.0 0.0 0.0 0.0 0.0 0.0 -0.0 0.1 0.1 0.0 0.0 -0.0 0.0 0.1 0.0 0.1 1.0 0.1 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.1 0.1 0.1 0.1 0.0 0.0 0.0 0.1 0.0
0.0 0.1 0.1 0.0 0.0 0.1 0.0 0.1 -0.1 -0.1 -0.1 -0.0 -0.1 -0.1 -0.1 0.3 0.1 1.0 -0.1 -0.1 0.1 -0.1 -0.1 0.0 -0.1 0.2 0.1 -0.1 0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1
0.1 0.1 0.1 0.0 0.0 0.1 0.0 0.1 0.5 0.4 0.5 -0.1 0.5 0.5 0.4 0.1 0.0 -0.1 1.0 0.5 0.1 0.5 0.4 0.2 0.2 -0.2 -0.1 0.4 -0.0 0.3 0.5 0.4 0.5 0.5 0.5
0.1 0.1 0.1 0.0 0.0 0.1 0.0 0.1 0.5 0.3 0.4 -0.1 0.4 0.5 0.3 0.1 0.0 -0.1 0.5 1.0 0.0 0.4 0.4 0.2 0.2 -0.2 -0.1 0.3 -0.0 0.3 0.4 0.3 0.4 0.4 0.4
0.0 0.1 0.1 0.0 0.0 0.0 0.0 0.0 0.1 0.1 0.1 -0.0 0.1 0.1 0.2 0.1 0.0 0.1 0.1 0.0 1.0 0.1 0.1 -0.0 0.1 0.1 0.1 0.1 0.0 0.1 0.1 0.1 0.1 0.2 0.2
0.1 0.1 0.1 0.0 0.0 0.1 0.0 0.1 0.6 0.5 0.7 -0.2 0.6 0.6 0.5 0.1 0.0 -0.1 0.5 0.4 0.1 1.0 0.6 0.2 0.3 -0.2 -0.1 0.5 -0.0 0.4 0.6 0.5 0.6 0.6 0.6
0.1 0.1 0.1 0.0 0.0 0.0 0.0 0.1 0.5 0.4 0.6 -0.1 0.6 0.5 0.5 0.1 -0.0 -0.1 0.4 0.4 0.1 0.6 1.0 0.1 0.2 -0.2 -0.1 0.4 -0.0 0.3 0.5 0.5 0.6 0.6 0.5
0.0 0.1 0.1 0.0 0.0 0.0 0.0 0.1 0.2 0.1 0.1 -0.1 0.1 0.2 0.1 0.0 0.0 0.0 0.2 0.2 -0.0 0.2 0.1 1.0 0.1 0.0 0.0 0.1 0.0 0.0 0.1 0.1 0.1 0.1 0.1
0.1 0.1 0.0 0.1 -0.0 -0.0 0.0 0.1 0.3 0.3 0.3 -0.0 0.3 0.3 0.3 0.1 0.0 -0.1 0.2 0.2 0.1 0.3 0.2 0.1 1.0 0.1 0.3 0.4 0.0 0.4 0.3 0.3 0.3 0.3 0.3
-0.1 -0.0 -0.0 0.0 -0.0 -0.0 0.0 -0.0 -0.2 -0.1 -0.2 0.1 -0.2 -0.2 -0.2 -0.0 0.0 0.2 -0.2 -0.2 0.1 -0.2 -0.2 0.0 0.1 1.0 0.6 -0.2 0.1 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2
-0.1 -0.0 -0.0 -0.0 -0.0 -0.0 0.0 -0.0 -0.1 -0.1 -0.1 -0.0 -0.1 -0.1 -0.1 -0.0 0.1 0.1 -0.1 -0.1 0.1 -0.1 -0.1 0.0 0.3 0.6 1.0 -0.0 0.2 -0.0 -0.1 -0.1 -0.1 -0.1 -0.1
0.1 0.1 0.1 0.0 0.0 0.0 -0.0 0.1 0.5 0.4 0.5 -0.1 0.4 0.5 0.5 0.1 0.1 -0.1 0.4 0.3 0.1 0.5 0.4 0.1 0.4 -0.2 -0.0 1.0 0.0 0.6 0.5 0.4 0.4 0.5 0.4
0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 -0.0 -0.0 -0.0 0.0 -0.1 -0.0 -0.0 0.1 0.1 0.1 -0.0 -0.0 0.0 -0.0 -0.0 0.0 0.0 0.1 0.2 0.0 1.0 0.1 -0.0 -0.0 -0.0 -0.0 -0.0
0.1 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.4 0.4 0.4 -0.1 0.4 0.4 0.4 0.1 0.1 -0.1 0.3 0.3 0.1 0.4 0.3 0.0 0.4 -0.2 -0.0 0.6 0.1 1.0 0.4 0.3 0.4 0.4 0.4
0.1 0.1 0.1 0.0 0.1 0.1 0.0 0.1 0.5 0.4 0.6 -0.1 0.5 0.5 0.6 0.1 0.0 -0.1 0.5 0.4 0.1 0.6 0.5 0.1 0.3 -0.2 -0.1 0.5 -0.0 0.4 1.0 0.6 0.6 0.7 0.7
0.1 0.1 0.1 0.0 0.0 0.0 0.0 0.0 0.4 0.4 0.5 -0.1 0.5 0.4 0.5 0.1 0.0 -0.1 0.4 0.3 0.1 0.5 0.5 0.1 0.3 -0.2 -0.1 0.4 -0.0 0.3 0.6 1.0 0.5 0.6 0.6
0.1 0.1 0.1 0.0 0.0 0.1 0.0 0.0 0.5 0.5 0.6 -0.1 0.6 0.5 0.5 0.1 0.0 -0.1 0.5 0.4 0.1 0.6 0.6 0.1 0.3 -0.2 -0.1 0.4 -0.0 0.4 0.6 0.5 1.0 0.6 0.6
0.1 0.1 0.1 0.0 0.0 0.1 0.0 0.1 0.5 0.5 0.6 -0.1 0.6 0.5 0.6 0.1 0.1 -0.1 0.5 0.4 0.2 0.6 0.6 0.1 0.3 -0.2 -0.1 0.5 -0.0 0.4 0.7 0.6 0.6 1.0 0.7
0.1 0.1 0.1 0.0 0.0 0.0 0.0 0.1 0.5 0.4 0.6 -0.1 0.5 0.5 0.6 0.1 0.0 -0.1 0.5 0.4 0.2 0.6 0.5 0.1 0.3 -0.2 -0.1 0.4 -0.0 0.4 0.7 0.6 0.6 0.7 1.0
Cond. Corr. given y=1
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
(a) Correct response (y = 1).
Qwen/Qwen2.5-72B-Instruct_verdicts
Mixtral-8x22B-Instruct-v0.1_verdicts
Meta-Llama-3.1-405B-Instruct-quantized.w8a16_verdicts
DeepSeekLlama70B_verdicts
DeepSeekQwen32B_verdicts
Llama-3.3-70B-Instruct_verdicts
SkyT1_verdicts
WizardLM-2-8x22B_verdicts
GRM_scores
URM_scores
QRM_scores
GPM_scores
GRMLlama32_scores
OffsetBias_scores
GRMGemma_scores
QwenPRM_min_scores
QwenPRM_max_scores
QwenPRM_avg_scores
InternLM2RewardModel_scores
InternLM2Reward7B_scores
Qwen72B_scores
DecisionTreeReward8B_scores
Skyworks_scores
ArmorRM_scores
EurusPRMStage1_min_scores
EurusPRMStage1_max_scores
EurusPRMStage1_avg_scores
EurusPRMStage2_min_scores
EurusPRMStage2_max_scores
EurusPRMStage2_avg_scores
QRMGemma_scores
LDLRewardGemma_scores
INFORM_scores
SkyworksGemma_scores
DecisionTreeReward27B_scores
1.0 0.2 0.1 0.0 0.1 0.2 0.2 0.0 0.1 0.0 0.1 -0.0 0.1 0.1 0.1 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.0 0.0 -0.0 -0.0 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1
0.2 1.0 0.1 0.1 0.1 0.3 0.1 0.1 0.1 0.0 0.1 -0.0 0.1 0.1 0.1 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.0 0.0 -0.0 0.0 0.1 0.0 0.0 0.1 0.1 0.1 0.1 0.1
0.1 0.1 1.0 0.0 0.0 0.1 0.1 0.0 0.0 -0.0 0.0 -0.0 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.0
0.0 0.1 0.0 1.0 0.0 0.1 0.0 0.0 0.0 -0.0 -0.0 0.0 -0.0 0.0 -0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 -0.0 -0.0 -0.0 0.0 0.0 0.0 -0.0 -0.0 0.0 0.0 0.0 0.0 0.0
0.1 0.1 0.0 0.0 1.0 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.0 0.0
0.2 0.3 0.1 0.1 0.0 1.0 0.1 0.1 0.1 0.0 0.1 0.0 0.1 0.1 0.0 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.0 0.0 -0.0 0.0 0.1 -0.0 0.1 0.1 0.1 0.1 0.1 0.1
0.2 0.1 0.1 0.0 0.0 0.1 1.0 0.0 0.1 0.0 0.1 0.0 0.0 0.1 0.0 0.1 -0.0 0.1 0.1 0.1 0.0 0.1 0.1 0.0 0.0 -0.0 -0.0 0.0 0.0 0.0 0.1 0.1 0.1 0.1 0.1
0.0 0.1 0.0 0.0 0.0 0.1 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.1 0.1 0.0 0.0 0.0 0.1 0.1 0.0 1.0 0.2 0.4 -0.0 0.4 0.5 0.4 0.1 0.0 -0.1 0.4 0.4 0.1 0.5 0.4 0.1 0.1 -0.1 -0.1 0.3 -0.0 0.2 0.5 0.3 0.5 0.4 0.4
0.0 0.0 -0.0 -0.0 -0.0 0.0 0.0 0.0 0.2 1.0 0.3 0.0 0.2 0.2 0.2 0.0 -0.0 -0.0 0.2 0.2 0.0 0.3 0.3 0.0 0.0 -0.1 -0.0 0.1 -0.0 0.1 0.2 0.2 0.3 0.2 0.2
0.1 0.1 0.0 -0.0 0.0 0.1 0.1 0.0 0.4 0.3 1.0 0.0 0.5 0.4 0.4 0.1 0.0 -0.1 0.4 0.4 0.1 0.6 0.5 0.0 0.1 -0.1 -0.1 0.2 -0.0 0.3 0.5 0.4 0.6 0.5 0.5
-0.0 -0.0 -0.0 0.0 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 1.0 0.0 0.0 -0.0 0.0 -0.0 0.0 -0.0 0.0 -0.0 -0.0 -0.0 0.0 0.0 -0.0 0.0 0.0 -0.0 -0.0 0.0 0.0 0.0 0.0 0.0
0.1 0.1 0.0 -0.0 0.0 0.1 0.0 0.0 0.4 0.2 0.5 0.0 1.0 0.4 0.5 0.1 -0.0 -0.1 0.4 0.3 0.1 0.5 0.5 0.0 0.1 -0.1 -0.1 0.2 -0.0 0.2 0.4 0.4 0.5 0.5 0.4
0.1 0.1 0.0 0.0 0.1 0.1 0.1 0.0 0.5 0.2 0.4 0.0 0.4 1.0 0.3 0.1 0.0 -0.1 0.4 0.4 0.1 0.4 0.4 0.1 0.1 -0.1 -0.1 0.3 -0.0 0.3 0.4 0.3 0.5 0.4 0.4
0.1 0.1 0.0 -0.0 0.0 0.0 0.0 0.0 0.4 0.2 0.4 -0.0 0.5 0.3 1.0 0.1 0.0 -0.1 0.3 0.3 0.2 0.4 0.4 -0.0 0.1 -0.1 -0.1 0.2 0.0 0.3 0.5 0.4 0.5 0.5 0.5
0.1 0.1 0.0 0.0 0.0 0.1 0.1 0.0 0.1 0.0 0.1 0.0 0.1 0.1 0.1 1.0 0.0 0.3 0.0 0.0 0.1 0.1 0.1 0.0 0.0 -0.0 -0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
0.0 0.0 -0.0 0.0 0.0 0.0 -0.0 -0.0 0.0 -0.0 0.0 -0.0 -0.0 0.0 0.0 0.0 1.0 0.1 -0.0 -0.0 0.0 0.0 0.0 -0.0 -0.0 0.0 0.0 0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.0
0.1 0.1 0.0 0.0 0.0 0.1 0.1 0.0 -0.1 -0.0 -0.1 0.0 -0.1 -0.1 -0.1 0.3 0.1 1.0 -0.1 -0.0 0.0 -0.1 -0.0 0.1 -0.0 0.1 0.1 -0.0 0.1 -0.0 -0.1 -0.1 -0.1 -0.1 -0.1
0.1 0.1 0.0 -0.0 0.0 0.1 0.1 0.0 0.4 0.2 0.4 -0.0 0.4 0.4 0.3 0.0 -0.0 -0.1 1.0 0.5 0.0 0.4 0.3 0.1 0.1 -0.1 -0.1 0.2 -0.0 0.2 0.4 0.3 0.5 0.4 0.4
0.1 0.1 0.0 0.0 0.0 0.1 0.1 0.0 0.4 0.2 0.4 0.0 0.3 0.4 0.3 0.0 -0.0 -0.0 0.5 1.0 0.0 0.4 0.3 0.1 0.1 -0.1 -0.1 0.2 -0.0 0.2 0.4 0.3 0.4 0.3 0.4
0.1 0.1 0.0 0.0 0.0 0.1 0.0 0.0 0.1 0.0 0.1 -0.0 0.1 0.1 0.2 0.1 0.0 0.0 0.0 0.0 1.0 0.1 0.1 -0.0 0.0 0.1 0.1 0.1 0.0 0.0 0.1 0.1 0.1 0.1 0.2
0.1 0.1 0.0 0.0 0.0 0.1 0.1 0.0 0.5 0.3 0.6 -0.0 0.5 0.4 0.4 0.1 0.0 -0.1 0.4 0.4 0.1 1.0 0.5 0.0 0.1 -0.1 -0.1 0.2 -0.0 0.2 0.5 0.4 0.6 0.5 0.5
0.1 0.1 0.0 -0.0 0.0 0.1 0.1 0.0 0.4 0.3 0.5 -0.0 0.5 0.4 0.4 0.1 0.0 -0.0 0.3 0.3 0.1 0.5 1.0 0.0 0.1 -0.1 -0.1 0.2 -0.0 0.2 0.4 0.4 0.5 0.5 0.4
0.0 0.0 0.0 -0.0 0.0 0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.0 0.1 -0.0 0.0 -0.0 0.1 0.1 0.1 -0.0 0.0 0.0 1.0 -0.0 0.1 0.1 -0.0 0.0 -0.0 0.0 -0.0 0.0 0.0 0.0
0.0 0.0 0.0 -0.0 0.0 0.0 0.0 -0.0 0.1 0.0 0.1 0.0 0.1 0.1 0.1 0.0 -0.0 -0.0 0.1 0.1 0.0 0.1 0.1 -0.0 1.0 0.1 0.1 0.2 0.1 0.2 0.1 0.1 0.1 0.1 0.1
-0.0 -0.0 0.0 0.0 0.0 -0.0 -0.0 -0.0 -0.1 -0.1 -0.1 -0.0 -0.1 -0.1 -0.1 -0.0 0.0 0.1 -0.1 -0.1 0.1 -0.1 -0.1 0.1 0.1 1.0 0.7 -0.1 0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1
-0.0 0.0 0.0 0.0 0.0 0.0 -0.0 0.0 -0.1 -0.0 -0.1 0.0 -0.1 -0.1 -0.1 -0.0 0.0 0.1 -0.1 -0.1 0.1 -0.1 -0.1 0.1 0.1 0.7 1.0 -0.1 0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1
0.1 0.1 0.0 0.0 0.0 0.1 0.0 0.0 0.3 0.1 0.2 0.0 0.2 0.3 0.2 0.1 0.0 -0.0 0.2 0.2 0.1 0.2 0.2 -0.0 0.2 -0.1 -0.1 1.0 0.1 0.4 0.3 0.3 0.3 0.3 0.3
0.0 0.0 0.0 -0.0 0.0 -0.0 0.0 0.0 -0.0 -0.0 -0.0 -0.0 -0.0 -0.0 0.0 0.1 0.0 0.1 -0.0 -0.0 0.0 -0.0 -0.0 0.0 0.1 0.1 0.1 0.1 1.0 0.1 -0.0 -0.0 0.0 0.0 -0.0
0.1 0.0 0.0 -0.0 -0.0 0.1 0.0 0.0 0.2 0.1 0.3 -0.0 0.2 0.3 0.3 0.1 0.0 -0.0 0.2 0.2 0.0 0.2 0.2 -0.0 0.2 -0.1 -0.1 0.4 0.1 1.0 0.3 0.3 0.3 0.3 0.3
0.1 0.1 0.1 0.0 0.0 0.1 0.1 0.0 0.5 0.2 0.5 0.0 0.4 0.4 0.5 0.1 -0.0 -0.1 0.4 0.4 0.1 0.5 0.4 0.0 0.1 -0.1 -0.1 0.3 -0.0 0.3 1.0 0.5 0.6 0.6 0.7
0.1 0.1 0.0 0.0 0.0 0.1 0.1 0.0 0.3 0.2 0.4 0.0 0.4 0.3 0.4 0.1 0.0 -0.1 0.3 0.3 0.1 0.4 0.4 -0.0 0.1 -0.1 -0.1 0.3 -0.0 0.3 0.5 1.0 0.4 0.5 0.5
0.1 0.1 0.0 0.0 0.0 0.1 0.1 0.0 0.5 0.3 0.6 0.0 0.5 0.5 0.5 0.1 0.0 -0.1 0.5 0.4 0.1 0.6 0.5 0.0 0.1 -0.1 -0.1 0.3 0.0 0.3 0.6 0.4 1.0 0.6 0.6
0.1 0.1 0.0 0.0 0.0 0.1 0.1 0.0 0.4 0.2 0.5 0.0 0.5 0.4 0.5 0.1 0.0 -0.1 0.4 0.3 0.1 0.5 0.5 0.0 0.1 -0.1 -0.1 0.3 0.0 0.3 0.6 0.5 0.6 1.0 0.6
0.1 0.1 0.0 0.0 0.0 0.1 0.1 0.0 0.4 0.2 0.5 0.0 0.4 0.4 0.5 0.1 0.0 -0.1 0.4 0.4 0.2 0.5 0.4 0.0 0.1 -0.1 -0.1 0.3 -0.0 0.3 0.7 0.5 0.6 0.6 1.0
Cond. Corr. given y=-1
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
(b) Incorrect response (y = −1).
Figure 4. Average conditional correlations in MMLU-Pro data based on model verdicts and scores for (a) correct responses (y = 1) and
(b) incorrect responses (y = −1).
16

## Page 17

FUSE: Ensembling Verifiers with Zero Labeled Data
Your response should look like this:
‘‘‘
{{thinking}}
<answer>
{{your_answer_here}}
</answer>
‘‘‘
The following prompt was adapted from Saad-Falcon et al. (2025b) for evaluation of extracted short-answer solutions against
ground truth. For multiple choice questions, answers were instead deterministically parsed.
Example evaluation prompt
Compare the following solutions to the given problem and determine
if they are equivalent. Return True only if the solutions are equivalent.
Solutions are short response, and may include mathematical expressions.
You should carefully check for semantic equivalence.
For example, ’One-half’ and ’\\frac{{10}}{{20}}’ are equivalent.
Solution 1:
{extracted}
Solution 2:
{ground_truth}
Enclose your final verdict as : <verdict>VERDICT</verdict>.
For example, if True, write <verdict>True</verdict>.
The following is a sample prompt used for verification generation given a sample response and the original question query.
We modified prompts for each verifier model to adapt to its capabilities and idiosyncrasies.
Example verification prompt
You are a strict auditor for technical tasks (math/science/coding/logic).
You get a problem and a candidate solution.
Score correctness WITHOUT any ground-truth.
GOAL: minimize false positives. Assume the solution is wrong until proven
correct. Do NOT give benefit-of-the-doubt.
Only reward what you can independently confirm from the problem.
Process (must follow):
1) Extract the FINAL ANSWER (last clearly committed result). If
ambiguous/conflicting -> NOT VERIFIED.
2) Independently check it: do at least ONE concrete verification step
(re-derive a key equation, test a case, check units, run through logic, etc.).
If you cannot perform a real check -> NOT VERIFIED.
3) Decide VERIFIED only if your check(s) confirm the final answer.
Hard constraints:
17

## Page 18

FUSE: Ensembling Verifiers with Zero Labeled Data
- Scores 4 or 5 are ONLY allowed if VERIFIED.
- If NOT VERIFIED, score MUST be <=3 (and usually 2 unless strong partial
progress).
- If you find a fatal flaw or the final answer is wrong -> score in {0,1,2}.
- If uncertain, choose the LOWER score.
Rubric (0-5):
5 = VERIFIED final answer; reasoning sound/complete; no meaningful gaps.
4 = VERIFIED final answer; reasoning has gaps/mistakes but answer still correct.
3 = NOT VERIFIED, but strong partial progress + multiple correct key steps;
close to verifiable.
2 = Some correct ideas, but major gaps/errors OR cannot justify final answer.
1 = Mostly wrong; minimal relevant correctness.
0 = Non-solution/irrelevant/refusal/nonsense.
Output format:
1) Brief analysis that begins with ‘VERIFIED’ or ‘NOT VERIFIED’
and mentions your concrete check.
2) New line at end: <score>X</score> where X is 0..5. Nothing after </score>."
Problem:
‘‘‘
{query}
‘‘‘
Candidate Solution:
‘‘‘
{generated responses}
‘‘‘
Verify the solution as best you can from the problem.
Then output the final score as <score>X</score>.
E.4. Experiments on Humanity’s Last Exam
This experimental setting involves generation of responses and verifications through the Google Gemini API, OpenAI API,
and DeepSeek API in addition to local generation using open-source models.
• Benchmark A subsample of Humanity’s Last Exam. As many verifiers do not natively support multi-modal input, we
first removed 394 questions requiring multi-modal input from the 2477 total available at cais/hle-rolling on
HuggingFace. A batch API request for 100 responses per question from Gemini-3-pro-preview was then submitted
with the following sampling parameters: temperature = 1.0, top p = 0.95, and a token limit of 30,000
per response. We then obtained our final sample by taking the first 50 responses from each question with at least 50
successful responses and at least one correct response (649 total; 181 multiple choice, 468 exact match). A notable
feature of this benchmark is its coverage of a wide range of topics ranging from more standard subjects such as
mathematics to fields in the humanities such as dance and literature. See Table 7 for a categorical breakdown of the
subsample, where we use the categories defined in the original HuggingFace dataset cais/hle-rolling.
• Generator model Gemini 3 Pro Preview
• Solution extraction model: Qwen3-Next-80B-A3B-instruct
• Verifier models
18

## Page 19

FUSE: Ensembling Verifiers with Zero Labeled Data
– Qwen2.5-72B-Instruct
– Skywork-Critic-Llama-3.1-70B
– gpt-oss-120b (no Harmony response format7)
– Gemini 3 Flash Preview
– DeepSeek-V3.2
– GPT-5.2 (high reasoning)
– GPT-5 mini (high reasoning)
• Number of Generations per Question 50
The pooled conditional correlation in this data (computed on all pooled solutions, conditional on the ground-truth label; thus
effectively weighted by label frequency) are shown in Figure 5. We observe positive correlations among verifiers with two
main blocks: (1) two weaker open-source models Qwen2.5-72B-Instruct and Skywork-Critic-Llama-3.1-70B; and (2) four
stronger models gpt-oss-120b, DeepSeek-V3.2, GPT-5.2 (high reasoning), and GPT-5 mini (high reasoning).
We impute missing rubric scores as 0 (i.e. verifier identifies the response as fully incorrect). Across the seven verifier
models, 8.63% of scores are missing, primarily due to batch API call failures, safety-related refusals/flags, and length or
context-window constraints.
E.5. Experiments on IMO Shortlist
All data for this experiment were generated from open-source models.
• Benchmark The IMO Shortlist subset of IMO AnswerBench (Luong et al., 2025). This 123-question benchmark
consists of modified versions of past problems in the IMO Shortlist. The modification, which is done by experts, helps
avoid memorization. See Table 8 for a categorical breakdown of IMO Shortlist subset, based on the standard categories
of Algebra, Combinatorics, Geometry, and Number Theory.
• Generator model Qwen3-30B-A3B-Thinking-2507
• Solution extraction model DeepSeek-R1-Distill-Llama-70B
• Verifier models
– DeepSeek-R1-Distill-Qwen-32B
– Kimi-Linear-48B-A3B-Instruct
– Llama-3.3-70B-Instruct
– Ministral-3-14B-Reasoning-2512
– Ministral-3-8B-Instruct-2512
– NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
– Qwen3-30B-A3B-Thinking-2507
– gemma-3-27b-it
– gpt-oss-20b
• Number of generations per question 50
The average conditional correlation in this data (conditional correlations weighted by label frequency) are uniformly mild
and positive (see Figure 6). Further, verifier balanced accuracies are, with a small exception in gpt-oss-20b, homogeneous
and only slightly better than random (see Figure 7).
We impute missing rubric scores as 0 (i.e. verifier identifies the response as fully incorrect). Across the nine verifier models,
4.15% of scores are missing, primarily due to length or context-window constraints.
7See https://developers.openai.com/cookbook/articles/openai-harmony. Interestingly, we found that omit-
ting Harmony slightly improves verification quality on ultra-hard tasks.
19

## Page 20

FUSE: Ensembling Verifiers with Zero Labeled Data
Qwen2.5-72B-Instruct_scores
Skywork-Critic-Llama-3.1-70B_scores
deepseek_reasoner_scores
gemini-3-flash_scores
gpt-oss-120b_scores
gpt5.2-high_scores
gptmini-high_scores
1.00
0.21
0.10
-0.06
0.14
0.16
0.17
0.21
1.00
0.15
-0.03
0.10
0.11
0.14
0.10
0.15
1.00
-0.04
0.29
0.39
0.38
-0.06
-0.03
-0.04
1.00
-0.04
-0.07
-0.04
0.14
0.10
0.29
-0.04
1.00
0.33
0.43
0.16
0.11
0.39
-0.07
0.33
1.00
0.51
0.17
0.14
0.38
-0.04
0.43
0.51
1.00
Pooled Corr given y=1
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
(a) Correct response (y = 1).
Qwen2.5-72B-Instruct_scores
Skywork-Critic-Llama-3.1-70B_scores
deepseek_reasoner_scores
gemini-3-flash_scores
gpt-oss-120b_scores
gpt5.2-high_scores
gptmini-high_scores
1.00
0.35
0.19
-0.04
0.21
0.16
0.18
0.35
1.00
0.23
-0.02
0.28
0.18
0.23
0.19
0.23
1.00
-0.05
0.37
0.42
0.47
-0.04
-0.02
-0.05
1.00
-0.05
-0.04
-0.05
0.21
0.28
0.37
-0.05
1.00
0.40
0.49
0.16
0.18
0.42
-0.04
0.40
1.00
0.53
0.18
0.23
0.47
-0.05
0.49
0.53
1.00
Pooled Corr given y=-1
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
(b) Incorrect response (y = −1).
Figure 5. Pooled correlations in HLE data conditional on (a) correct responses (y = 1) and (b) incorrect responses (y = −1). Raw scores
are used without normalization or binarization.
20

## Page 21

FUSE: Ensembling Verifiers with Zero Labeled Data
DeepSeek-R1-Distill-Qwen-32B
Kimi-Linear-48B-A3B-Instruct
Llama-3.3-70B-Instruct
Ministral-3-14B-Reasoning-2512
Ministral-3-8B-Instruct-2512
NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
Qwen3-30B-A3B-Thinking-2507
gemma-3-27b-it
gpt-oss-20b
1.00
0.09
0.13
0.17
0.17
0.09
0.14
0.13
0.09
0.09
1.00
0.12
0.11
0.13
0.12
0.18
0.07
0.09
0.13
0.12
1.00
0.21
0.20
0.17
0.22
0.18
0.20
0.17
0.11
0.21
1.00
0.28
0.15
0.26
0.22
0.16
0.17
0.13
0.20
0.28
1.00
0.16
0.27
0.23
0.18
0.09
0.12
0.17
0.15
0.16
1.00
0.21
0.10
0.19
0.14
0.18
0.22
0.26
0.27
0.21
1.00
0.23
0.20
0.13
0.07
0.18
0.22
0.23
0.10
0.23
1.00
0.12
0.09
0.09
0.20
0.16
0.18
0.19
0.20
0.12
1.00
Average Correlation Matrix
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
Figure 6. Expected conditional correlations of the verifiers given response correctness averaged over all responses (i.e. (i, j)th entry is
corr(vi, vj|y = 1)p(y = 1) + corr(vi, vj|y = −1)p(y = −1)) in IMO Shortlist data. Verifier scores are used without normalization or
binarization.
E.6. Mixed data ablation
All data for our mixed data ablations are from Saad-Falcon et al. (2025a). As in our main experiments, we make no
modification to either the raw data or the verifier ensemble.
E.7. Repeated verification is redundant
Let vj(ri) denote i.i.d verifications of response ri. When verifications are i.i.d, the optimal ensemble is a naive ensemble, so
the induced selection rule is the following:
i⋆= arg max
i
1
m
m
X
j=1
vj(ri).
Observe that for any m, by linearity of expectation, the naive ensemble 1
m
Pm
j=1 vj(ri) has the same sensitivity, specificity,
and balanced accuracy as any individual vj(rj). On the other hand, as m →∞, the above selection rule converges to
i⋆= arg maxi E[vj(ri)]. Scaling repeated verifications therefore simply replaces a stochastic verifier with a deterministic
equivalent that has identical balanced accuracy. While balanced accuracy and selection accuracy are distinct, as the former
is not sufficient to determine the latter, this explains the observation by Saad-Falcon et al. (2025b) that sampling five times
from a strong verifier typically performs identically to sampling merely once (see their Table 21).
E.8. Additional unsupervised baselines
In the main text, we focus on majority vote and naive ensemble as unsupervised baselines as these are the most commonly
used methods in the language model setting. Indeed, we are unaware of any prior work that connects the literature on
unsupervised ensemble learning to test-time scaling. In Table 9, we report the performance of existing unsupervised
baselines on the Saad-Falcon et al. (2025b) data. These are:
• Dawid & Skene (1979): An Expectation-Maximization algorithm that assumes conditional independence.
• Jaffe et al. (2016): An extension of Jaffe et al. (2015) to settings with structured conditional dependence.
• Gaussian mixture model: We assume that conditional on the binary label yi, verifier scores are jointly multivariate
Gaussian and estimate parameters through the EM algorithm.
21

## Page 22

FUSE: Ensembling Verifiers with Zero Labeled Data
0.0
0.2
0.4
0.6
0.8
1.0
Balanced Accuracy
DeepSeek-R1-Distill-Qwen-32B
Kimi-Linear-48B-A3B-Instruct
Llama-3.3-70B-Instruct
Ministral-3-14B-Reasoning-2512
Ministral-3-8B-Instruct-2512
NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
Qwen3-30B-A3B-Thinking-2507
gemma-3-27b-it
gpt-oss-20b
Verifier balanced accuracy
Figure 7. Balanced accuracies of verifiers on IMO Shortlist data.
These numbers uniformly fall below those of FUSE in Table 1, indicating that our performance is not a mere consequence
of recognizing that the unsupervised ensembling literature has relevance to LLM verification.
F. Compute requirements
Verifications by non-API models for Humanity’s Last Exam and all responses and verifications for IMO AnswerBench were
generated locally on a compute node with 8 NVIDIA H100 GPUs with 80 GB of memory each. All other aspects of this
work, including the experiments on data from Saad-Falcon et al. (2025a), did not require GPU usage or other forms of
specialized compute.
G. Replicability
We open-source our raw HLE and IMO Shortlist data at https://huggingface.co/FUSE-verifiers. Our
supplementary material contains instructions for running the replication code of Saad-Falcon et al. (2025a), which we use
to generate the Weaver numbers in Table 1. We will open-source a detailed version of our experimental pipeline (data
generation, evaluation, etc.).
22

## Page 23

FUSE: Ensembling Verifiers with Zero Labeled Data
Name
Type
8B
70B
DeepSeekLlama70B
binary
No
Yes
DeepSeekQwen32B
binary
No
Yes
Llama-3.3-70B-Instruct
binary
No
Yes
Meta-Llama-3.1-405B-Instruct-quantized.w8a16
binary
No
Yes
Mixtral-8x22B-Instruct-v0.1
binary
No
Yes
Qwen/Qwen2.5-72B-Instruct
binary
No
Yes
SkyT1
binary
No
Yes
WizardLM-2-8x22B
binary
No
Yes
ArmorRM
reward model
Yes
Yes
DecisionTreeReward27B
reward model
No
Yes
DecisionTreeReward8B
reward model
No
Yes
EurusPRMStage1 avg
reward model
Yes
No
EurusPRMStage1 max
reward model
Yes
Yes
EurusPRMStage1 min
reward model
No
Yes
EurusPRMStage2 avg
reward model
No
Yes
EurusPRMStage2 max
reward model
Yes
No
EurusPRMStage2 min
reward model
Yes
Yes
GPM
reward model
Yes
Yes
GRMGemma
reward model
Yes
Yes
GRMLlama32
reward model
Yes
No
GRM
reward model
Yes
Yes
INFORM
reward model
No
Yes
InternLM2Reward7B
reward model
Yes
Yes
InternLM2RewardModel
reward model
No
Yes
LDLRewardGemma
reward model
No
Yes
OffsetBias
reward model
Yes
Yes
QRMGemma
reward model
No
Yes
QRM
reward model
Yes
Yes
Qwen72B
reward model
No
Yes
QwenPRM avg
reward model
Yes
Yes
QwenPRM max
reward model
No
Yes
QwenPRM min
reward model
Yes
Yes
SkyworksGemma
reward model
No
Yes
Skyworks
reward model
Yes
Yes
URM
reward model
Yes
Yes
Table 6. Verifiers in Saad-Falcon et al. (2025b) experimental setting
Table 7. Category summary for HLE subset.
Category
Count
Math
141
Humanities/Social Science
110
Other
87
Computer Science/AI
84
Biology/Medicine
77
Physics
77
Chemistry
48
Engineering
18
Chess/Logic/Puzzle
7
23

## Page 24

FUSE: Ensembling Verifiers with Zero Labeled Data
Table 8. Category summary for IMO Shortlist subset.
Category
Count
Algebra
35
Combinatorics
45
Number Theory
35
Table 9. Best-of-100 selection accuracy of additional unsupervised baselines on Saad-Falcon et al. (2025b) data.
Size
Benchmark
DS
GMM
DCL
8B
GPQA
0.3168
0.3709
0.2832
GPQA Diamond
0.3295
0.4184
0.3279
MATH500
0.5513
0.6729
0.6804
MMLU
0.7179
0.8032
0.7910
MMLU-Pro
0.5394
0.6408
0.6199
70B
GPQA
0.5427
0.5670
0.5601
GPQA Diamond
0.5240
0.5413
0.5606
MATH500
0.8115
0.8726
0.8693
MMLU
0.8480
0.8952
0.9013
MMLU-Pro
0.7497
0.8263
0.8220
24
