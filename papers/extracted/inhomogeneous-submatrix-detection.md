---
source_pdf: papers/Inhomogeneous Submatrix Detection.pdf
slug: inhomogeneous-submatrix-detection
pages: 38
extracted_on: 2026-07-29
---

# Inhomogeneous Submatrix Detection

## Page 1

Inhomogeneous Submatrix Detection
Mor Oren-Loberman
Dvir Jerbi
Tamir Bendory
Wasim Huleihel
March 13, 2026
Abstract
In this paper, we study the problem of detecting multiple hidden submatrices in
a large Gaussian random matrix when the planted signal is inhomogeneous across
entries. Under the null hypothesis, the observed matrix has independent and identically
distributed standard normal entries. Under the alternative, there exist several planted
submatrices whose entries deviate from the background in one of two ways: in the mean-
shift model, planted entries (templates) have nonzero and possibly varying means; in
the variance-shift model, planted entries have inflated and possibly varying variances.
We consider two placement regimes for the planted submatrices.
In the first, the
row and column index sets are arbitrary. Motivated by scientific applications, in the
second regime the row and column indices are restricted to be consecutive. For both
alternatives and both placement regimes, we analyze the statistical limits of detection
by proving information-theoretic lower bounds and by designing algorithms that match
these bounds up to logarithmic factors, for a wide family of templates.
1
Introduction
This paper investigates the problem of detecting hidden submatrices embedded in a large
Gaussian random matrix. Under the null hypothesis, the observed n × n matrix consists
of independent and identically distributed standard normal entries. Under the alternative
hypothesis, there exist m disjoint submatrices of size k × k whose entries follow Gaussian
distributions with possibly non-uniform means or variances. The objective is to construct a
test, equivalently an algorithm, that reliably distinguishes between these two hypotheses.
We begin by considering two models for the placement of the planted submatrices.
In the first model, the planted blocks are disjoint and their row and column supports
may be arbitrary subsets of indices.
The detection and recovery versions of this for-
mulation correspond to the well-studied problems of submatrix detection and submatrix
All authors are with the Department of Electrical and Computer Engineering-Systems at Tel Aviv
University, Tel Aviv 6997801, Israel (e-mails: orenmor@mail.tau.ac.il, dvirjerbi@mail.tau.ac.il,
bendory@tauex.tau.ac.il, wasimh@tauex.tau.ac.il).
This work is supported by the ISRAEL SCI-
ENCE FOUNDATION (grant No. 1734/21). T.B. is supported in part by BSF under Grant 2020159, in
part by NSF-BSF under Grant 2024791, in part by ISF under Grant 1924/21, and in part by a grant from
The Center for AI and Data Science at Tel Aviv University (TAD).
1
arXiv:2603.09602v2  [math.ST]  11 Mar 2026

## Page 2

localization, which have attracted considerable attention in recent years; see, for exam-
ple, [SWP+09, KBRS11, BKR+11, BI13, ACV14, HWX15, MRZ15, VAC15, MW15, SN13,
ACCD10, BDN17, CX16, CLR17, BBH18, BBH19, Hul22, RHS24, EH25, Hul22] and the
references therein.
In the canonical setting of a single planted block, one seeks to determine whether an
n×n matrix sampled from a distribution Q contains a hidden k ×k submatrix whose entries
are drawn from a different distribution P. When both P and Q are Gaussian, the statistical
limits of detection, optimal testing procedures, and computational lower bounds have been
characterized in detail in [BI13, MRZ15, SWP+09, KBRS11, BKR+11, MW15, BBH19].
If instead P and Q are Bernoulli distributions, the problem reduces to the planted
dense subgraph model, which has also been extensively investigated; see, for example,
[BI13, ACV14, VAC15, HWX15, BBH18].
A central insight emerging from this line of
work, both in the Gaussian and Bernoulli settings, is the existence of a statistical com-
putational gap: the minimal signal size k required for information-theoretic detectabil-
ity can be strictly smaller than the minimal size for which detection is achievable by
polynomial-time algorithms. The recovery problem has likewise been studied extensively,
across different distributional assumptions and in both single and multiple block regimes;
see [CX16, Mon15, CC18, HWX16, HWX17, CLR17, BBH18]. These works characterize
when the support of the planted submatrix can be accurately identified and analyze both
information-theoretic and computational thresholds.
The arbitrary placement model described above is particularly natural in applications
such as biclustering. Biclustering methods aim to uncover structured submatrices, often
interpreted as hidden clusters, within large data tables of samples versus variables. Such
techniques arise in a variety of domains, including community detection in social networks
and the analysis of gene expression data from microarrays. Nevertheless, in certain scien-
tific and engineering contexts, assuming arbitrary index sets may be unrealistic. Motivated
by such considerations, we also study a structured placement model in which each planted
block is supported on consecutive rows and consecutive columns. A prominent example arises
in single-particle cryo-electron microscopy, a leading experimental technique for determin-
ing the three-dimensional structures of macromolecules, such as proteins [BMS15, Lyu19].
Early stages of the cryo-electron microscopy computational pipeline require locating nu-
merous particle images, which are noisy two-dimensional projections of randomly oriented
molecular copies, within a large noisy micrograph [Sin18, BBS20].
This step, known as
particle picking, is algorithmically challenging. While various heuristic and learning-based
approaches have been proposed, including [WGL+16, HAS18, BMR+19, ELS20, EWD+24],
a systematic investigation of the fundamental statistical and computational limits of this
detection problem remains largely unexplored.
Our work is most closely related to [DHB24], which analyzes detection and recovery of
multiple homogeneous Gaussian submatrices under both arbitrary and consecutive place-
ments and establishes sharp information-theoretic and computational thresholds. In that
setting, each planted block is homogeneous, meaning that its entries share a common dis-
tribution across the entire block. In contrast, the present work develops a more general
framework that allows structured heterogeneity within each planted submatrix. We intro-
duce a finite-template model in which every block is assigned one template from a fixed
finite collection, and the distribution of each entry depends on its relative coordinate within
2

## Page 3

the block and on the chosen template. The classical homogeneous model is recovered as a
special case when all templates coincide and are constant across coordinates. This gener-
alization is motivated by the observation that, in many realistic scenarios, signals are not
spatially uniform. Instead, they may exhibit gradients, anisotropies, or other structured
patterns that cannot be captured by a single mean or variance parameter. In particular,
allowing coordinate-dependent structure fundamentally changes both the statistical and an-
alytical landscape of the problem. The detectability threshold is no longer governed by a
single scalar shift, and the interaction between heterogeneous templates and random block
overlaps must be handled explicitly. Addressing these challenges requires new probabilistic
tools and a substantially more delicate second-moment Bayes risk analysis that captures
how coordinate-dependent signals interact through random block overlaps; phenomena that
are absent in the homogeneous setting. We now formalize this framework in the Gaussian
setting and summarize our main results.
We focus on two Gaussian variants of this framework: mean-shift templates and variance-
shift templates. In the mean-shift model, each template specifies a coordinate-wise mean
profile over the k×k block, while the noise variance remains fixed. In the variance-shift model,
the mean remains zero and the template assigns coordinate-wise variances. For both variants,
we analyze arbitrary, non-consecutive placements as well as consecutive placements in which
each block occupies a Cartesian product of row and column intervals of length k. These two
placement regimes differ in their combinatorial complexity and in the distribution of overlaps
between candidate blocks, leading to distinct detection thresholds. In the consecutive setting,
we additionally study a circular variant in which indices are taken modulo n. This restores
translation invariance and facilitates the lower-bound analysis without altering the essential
statistical behavior.
Our main results establish information-theoretic lower bounds for detection in the finite-
template model via a second-moment analysis of the likelihood ratio under the null hypothe-
sis. The resulting threshold is characterized by a scalar quantity determined by the entrywise
χ2-divergences associated with the templates and by the overlap distribution induced by the
placement family. Under a mild regularity condition that excludes highly concentrated tem-
plates, this quantity reduces to a natural signal energy parameter. On the algorithmic side,
we study several simple and computationally efficient test statistics, including global linear
and quadratic statistics as well as template-matched scan procedures. These tests succeed in
complementary parameter regimes. Under the same regularity condition, we identify param-
eter ranges in which these procedures match the information-theoretic detection boundary,
up to logarithmic factors.
Notation.
For any positive integer k, let [k] ≜{1, 2, . . . , k}. We write |U| for the car-
dinality of a finite set U, and 1 {E } for the indicator of an event E .
By convention,
1
|U|
P
u∈U g(u) = 1 when U = ∅.
For a matrix M ∈Rk×k, we use the standard norms
∥M∥1 ≜P
i,j∈[k] |Mij|, ∥M∥2
F ≜P
i,j∈[k] M 2
ij, and ∥M∥∞≜maxi,j∈[k] |Mij|.
We write N(µ, σ2) for a univariate Gaussian distribution with mean µ and variance σ2.
We write X ⊥⊥Y when random variables X and Y are independent. For probability measures
P ≪Q on a common measurable space, we write χ2(P∥Q) ≜
R
(dP/dQ −1)2dQ for the chi-
square divergence. We also write dTV(P, Q) ≜1
2
R
|dP −dQ| and dKL(P∥Q) ≜
R
dP log dP
dQ
for total variation and Kullback–Leibler divergence, respectively. For a matrix Σ ∈Rk×k
3

## Page 4

with entries satisfying 0 ≤Σij < 1, we define the blockwise Kullback–Leibler divergence
KL(Σ) ≜1
2
P
i,j∈[k] (Σij −log(1 + Σij)).
We use standard asymptotic notation: for sequences {an} and {bn}, we write an = O(bn),
an = Ω(bn), an = Θ(bn), an = o(bn), and an = ω(bn) with their usual meanings as n →∞.
All asymptotic statements are taken in the limit n →∞, with m = m(n) and k = k(n)
possibly depending on n unless stated otherwise. For a, b ∈R, we write a ∨b ≜max{a, b}
and a ∧b ≜min{a, b}. Throughout the paper, C denotes a positive constant whose value
may change from line to line and which is independent of n, m, k, and all signal parameters;
any dependence on fixed model objects, such as the template family, will be stated explicitly
when relevant.
2
Problem Formulation
In this section, we introduce and formulate the statistical models and detection problems,
analyzed in this paper.
Blocks, placements, and indexing.
Let m, k, n ∈N satisfy mk ≤n.
A block (or
submatrix) is a Cartesian product B = S ×T ⊂[n]×[n] where S, T ⊂[n] and |S| = |T | = k.
Note that the entries of S and T need not be consecutive integers, i.e., the block may be
scattered rather than contiguous. Let
Bk,n ≜{S × T : S, T ⊂[n], |S| = |T| = k} ,
(1)
be the family of all k ×k blocks, and define the family of unordered collections of m pairwise
disjoint blocks by
Kk,m,n ≜{K ⊂Bk,n : |K| = m, B ∩B′ = ∅∀B ̸= B′ ∈K} .
(2)
In the consecutive placement model, let Ccon
k,n ≜{{i + 1, . . . , i + k} : i = 0, . . . , n −k} and set
Bcon
k,n ≜

S × T : S, T ∈Ccon
k,n
	
,
(3)
Kcon
k,m,n ≜

K ⊂Bcon
k,n : |K| = m, B ∩B′ = ∅∀B ̸= B′ ∈K
	
.
(4)
In addition to the standard consecutive placement model, we also consider a circular consec-
utive variant, in which indices are taken modulo n. Define C◦
k,n ≜

{i + 1, . . . , i + k} mod n :
i = 0, . . . , n −1
	
, and set
B◦
k,n ≜

S × T : S, T ∈C◦
k,n
	
,
(5)
K◦
k,m,n ≜

K ⊂B◦
k,n : |K| = m, B ∩B′ = ∅∀B ̸= B′ ∈K
	
.
(6)
The circular consecutive model restores translation symmetry across block locations. Note
that |Bcon
k,n| = (n −k + 1)2, whereas |B◦
k,n| = n2. When both models are considered, results
for the circular consecutive placement are stated explicitly, with extensions to the standard
consecutive placement specified when applicable. Figure 1 illustrates the placement regimes
introduces above.
4

## Page 5

Kk,m,n
Kcon
k,m,n
Figure 1: Schematic illustration of the placement families (shown for n = 16, k = 4). In the
non-consecutive model, arbitrary row and column subsets are selected, yielding blocks of the form
S × T. In the consecutive model, the row and column sets are intervals of length k.
Finally, let B = S × T be a block with S = {s1 < · · · < sk} and T = {t1 < · · · < tk}.
We define the induced coordinate map
φB : S × T →[k] × [k],
φB(su, tv) = (u, v).
(7)
That is, φB assigns to each entry of B its relative row and column indices within the block.
We emphasize that the map φB is deterministic and not part of the generative model; it is an
indexing convention induced by the ordering of S and T . Its role is to align the entries of a
block located at arbitrary coordinates in [n]×[n] with a canonical k ×k template indexed by
[k] × [k]. We remark below why this alignment is essential in the inhomogeneous submatrix
model. Figure 2 illustrates the alignment induced by φB.
Finite-template submatrix detection model.
Throughout, we focus on two Gaussian
settings: submatrix detection under a mean shift and under a variance shift. For notational
convenience, we write Q ≜N(0, 1) for the standard normal distribution, and Q⊗n×n denotes
the corresponding i.i.d. product measure on Rn×n. Furthermore, we consider a finite family
of templates indexed by ℓ∈[m]. For each ℓ, let {Pℓ,u}u∈[k]×[k] be a collection of probability
distributions on R. The detection problem is to test
H0 : X ∼Q⊗n×n
vs.
H1 : X ∼D(n, k, m, {Pℓ,u}, Q),
(8)
where D(n, k, m, {Pℓ,u}, Q) is defined as follows. Under H1, an unordered collection K of
m pairwise disjoint blocks is drawn uniformly at random from a specified placement family:
Kk,m,n in the non-consecutive case, Kcon
k,m,n in the standard consecutive case, or K◦
k,m,n in the
circular consecutive case. Conditional on K, a bijection β : K →[m] is drawn uniformly at
random from all such bijections. Given (K, β), the entries of X are independent and satisfy
Xij ∼
(
Pβ(B), φB(i,j),
(i, j) ∈B for some B ∈K,
Q,
otherwise.
(9)
5

## Page 6

t1 t2
t3
t4
s1
s2
s3
s4
M1
M2
φB1(s2, t3) = (2, 3)
φB2(s2, t3) = (2, 3)
Figure 2: Illustration of the coordinate map φB in (7). For a block B = S × T, the map φB(i, j) =
(u, v) records the relative row and column indices of (i, j) within B, thereby aligning entries of B
with template coordinates. The figure shows both a consecutive block B1 and a non-consecutive
block B2 mapped to their respective templates.
We study two Gaussian specializations of the finite-template model.
• Mean-shift model. Let M = {Mℓ}m
ℓ=1 with Mℓ∈Rk×k. For (u, v) ∈[k] × [k], define
Pℓ,(u,v) = N ((Mℓ)uv, 1) .
(10)
• Variance-shift model. Let S = {Σℓ}m
ℓ=1 with Σℓ∈Rk×k
+
satisfying maxu,v(Σℓ)uv ≤ϑ0
for some fixed ϑ0 ∈[0, 1) that is independent of n. Define
Pℓ,(u,v) = N (0, 1 + (Σℓ)uv) .
(11)
Throughout the paper, we denote by PH0 the null distribution Q⊗n×n and by PH1 the mixture
alternative induced by the finite-template model under the placement family specified in each
statement. When necessary, we write Pcon
H1 and P◦
H1 to distinguish between the standard and
circular consecutive placement models.
Finally, we would like to remark why the alignment in (7) is essential in the inhomo-
geneous submatrix model. In contrast to the classical homogeneous setting, where all en-
tries of the planted submatrix share the same distribution (e.g., Gaussians with the same
mean as in [MW15, DHB24]), here the distributions may vary across positions within the
block. The signal is therefore specified in local block coordinates, say by a template matrix
M = (Mu,v)u,v∈[k]. The map φB allows us to embed this fixed template into any candi-
date block B by setting, for (i, j) ∈B, E[Xi,j] = MφB(i,j). Thus, φB provides a canonical
identification between global matrix coordinates and local block coordinates.
Goal.
Given an observation X, a detection algorithm An outputs a decision in {0, 1},
corresponding to the null and alternative hypotheses. The risk of An is defined as the sum
of its Type I and Type II error probabilities,
R(An) ≜PH0 (An(X) = 1) + PH1 (An(X) = 0) ,
(12)
6

## Page 7

where PH0 denotes the distribution of X under the null hypothesis and PH1 denotes the
marginal distribution of X under the alternative. In particular, PH1 is the mixture distribu-
tion obtained by averaging over the random choice of the planted block collection K and,
conditional on K, over the random labeling β : K →[m] specified by the model. Equivalently,
PH1(·) = EK,β

PH1|K,β(·)

.
(13)
We say that An solves the detection problem if R(An) →0 as n →∞. The procedures
considered in this paper are either unrestricted (and potentially computationally expensive)
or restricted to run in polynomial time.
3
Main Results
In this section, we present our main results on submatrix detection under the finite-template
model introduced in Section 2. We identify parameter regimes in which detection is possible
and those in which it is information-theoretically impossible. We also investigate detection
under polynomial-time computational constraints.
3.1
Upper bounds
3.1.1
Mean-shift model
We begin by establishing achievable detection guarantees for the finite-template mean-shift
model. To this end, we introduce the proposed detection algorithms and then state their
performance guarantees.
Global test.
We first consider a global test based on aggregating all matrix entries. Define
Tsum(X) ≜sign(µdet)
X
i,j∈[n]
Xij,
µdet ≜
m
X
ℓ=1
X
u,v∈[k]
(Mℓ)uv,
(14)
where µdet is the total planted mean mass contributed by all templates. Note that under
the model, each template appears exactly once among the planted blocks by construction,
so the total expected signal contribution equals µdet, independently of the block locations.
Throughout, we assume µdet ̸= 0. When µdet = 0, the global sum statistic carries no signal,
and we rely exclusively on scan-based procedures. Then, we define the global sum test as
Asum(X) ≜1 {Tsum(X) ≥τsum} .
(15)
where τsum ≜|µdet|
2 . We note that Tsum can be computed in linear time; thus, the sum test is
computationally efficient.
Scan test.
Let M ∈Rk×k be a fixed template matrix. For a given family of blocks B
(either Bk,n, Bcon
k,n or B◦
k,n), define the scan statistic
Tµ
scan(X; B, M) ≜max
B∈B
X
(i,j)∈B
MφB(i,j) Xij,
(16)
7

## Page 8

where φB is the induced coordinate map defined in (7). Note that (16) “scans” w.r.t. a
single template M only. Intuitively, under the alternative hypothesis, each planted block
is associated with a distinct template from the finite family M = {Mℓ}m
ℓ=1. For a fixed
block location B, the expected value of the scan statistic in (16) equals ∥Mℓ∥2
F, when the
planted template is Mℓ. Consequently, among the templates in M, the one with the largest
Frobenius norm yields the largest expected shift for this statistic. Since detection requires
only that at least one planted block produces a statistically significant excursion, it suffices
to scan with the template achieving the maximal Frobenius norm. We therefore define
ℓmax ∈arg max
ℓ∈[m] ∥Mℓ∥2
F ,
Mmax ≜Mℓmax.
(17)
Accordingly, the template-aware scan test over a family of blocks B is defined as
Aµ
scan,max(X; B) ≜1 {Tµ
scan(X; B, Mmax) ≥τ µ
scan(B)} ,
(18)
where
τ µ
scan(B) ≜





r
(4 + δ) ∥Mmax∥2
F k log en
k ,
B = Bk,n,
q
(4 + δ) ∥Mmax∥2
F log n,
B ∈{Bcon
k,n, B◦
k,n},
(19)
for a fixed constant δ > 0. Note that the global sum test and the scan test for the stan-
dard and circular consecutive placement families are computationally efficient (running in
polynomial time), whereas the scan test for the non-consecutive placement family is com-
putationally expensive (having exponential time complexity). More related discussions are
provided in Remark 2.
We are now in a position to state our main results.
Theorem 1 (Mean-shift upper bounds). Consider the finite-template mean-shift model in-
troduced in Section 2. Let Mmax be defined as in (17). The following statements hold.
1. If
|µdet| = ω(n),
(20)
then the global sum test Asum(X) in (15) satisfies R(Asum) = o(1) under the non-
consecutive placement regime and under both consecutive placement regimes.
2. If
∥Mmax∥2
F = ω

k log n
k

,
(21)
then the template-aware scan test Aµ
scan,max(X; Bk,n) in (18) satisfies R
 Aµ
scan,max

=
o(1).
3. If
∥Mmax∥2
F = ω(log n),
(22)
then the template-aware scan test Aµ
scan,max(X; B) in (18) satisfies R
 Aµ
scan,max

= o(1),
under the standard consecutive placement family Bcon
k,n; the same bound holds under the
circular consecutive placement family B◦
k,n.
8

## Page 9

3.1.2
Variance-shift model
We now move forward to the variance-shift detection model introduced in Section 2. As
in the mean-shift case, we consider both global and scan-based procedures. The resulting
scan statistic coincides with the blockwise log-likelihood ratio under the Gaussian variance
alternatives.
Global test.
Define the centered quadratic statistic
Tquad(X) ≜
X
i,j∈[n]
 X2
ij −1

.
(23)
We note that under the null hypothesis, EH0[Tquad(X)] = 0. Under the alternative, the mean
increases by the total variance mass contributed by the planted blocks, namely,
νdet ≜
m
X
ℓ=1
X
u,v∈[k]
(Σℓ)uv.
(24)
Accordingly, we define the global quadratic test as
Aquad(X) ≜1{Tquad(X) ≥τquad},
(25)
where τquad ≜νdet
2 .
Scan test.
Let Σ ∈Rk×k
+
be a fixed variance template satisfying maxu,v Σuv ≤ϑ0 for a
constant ϑ0 ∈[0, 1) independent of n. For a given family of blocks B (either Bk,n, Bcon
k,n or
B◦
k,n), define the scan statistic
Tσ
scan(X; B, Σ) ≜max
B∈B
X
(i,j)∈B
1
2

ΣφB(i,j)
1 + ΣφB(i,j)
X2
ij −log
 1 + ΣφB(i,j)

,
(26)
where φB is the induced coordinate map defined in (7). This statistic coincides with the
blockwise log-likelihood ratio for a Gaussian variance-shift template. For such a template Σ,
we further define its associated blockwise Kullback–Leibler divergence
KL(Σ) ≜1
2
X
u,v∈[k]
[(Σ)uv −log (1 + (Σ)uv)] .
(27)
Accordingly, we define the corresponding finite-template scan test as
Aσ
scan(X; B) ≜1

max
ℓ∈[m] Tσ
scan(X; B, Σℓ) ≥τ σ
scan(B)

,
(28)
where
τ σ
scan(B) ≜
(
(1 + δ)
 2k log en
k + log m

,
B = Bk,n,
(1 + δ) (2 log n + log m) ,
B ∈

Bcon
k,n, B◦
k,n
	
.
(29)
for a fixed constant δ > 0. We have the following result.
9

## Page 10

Theorem 2 (Variance-shift upper bounds). Consider the finite-template variance-shift
model introduced in Section 2. Then the following statements hold.
1. If
νdet = ω(n),
(30)
then the global quadratic test Aquad(X) in (25) satisfies R(Aquad) = o(1) under the
non-consecutive placement regime and under both consecutive placement regimes.
2. If
max
ℓ
KL(Σℓ) = ω

k log n
k + log m

,
(31)
then the finite-template scan test Aσ
scan(X; Bn,k) in (28) satisfies R (Aσ
scan) = o(1).
3. If
max
ℓ
KL(Σℓ) = ω(log n + log m),
(32)
then the finite-template scan test Aσ
scan(X; B) in (28) satisfies R (Aσ
scan) = o(1), under
the standard consecutive placement family Bcon
k,n; the same bound holds under the circular
consecutive placement family B◦
k,n.
Remark 1. In the mean-shift setting, the scan statistic is linear in the data, and its ex-
pectation under the alternative equals ∥Mℓ∥2
F when the planted template is Mℓ. Scanning
with a template attaining maxℓ∥Mℓ∥F therefore maximizes the expected value of this statistic
among the planted blocks. This formulation isolates the dependence of the detection threshold
on the block family B. Alternatively, one may scan jointly over block locations and templates,
defining
max
ℓ∈[m] max
B∈B
X
(i,j)∈B
(Mℓ)φB(i,j)Xij.
(33)
By a union bound over ℓ∈[m], this procedure achieves vanishing risk whenever
∥Mmax∥2
F = ω (log |B| + log m) ,
(34)
that is, under the same scaling as in Theorem 1 with the scan threshold increased by an
additive log m term.
In contrast, in the variance-shift setting the scan statistic coincides with a blockwise log-
likelihood ratio, and the relevant separation is quantified by the Kullback–Leibler divergence.
In this case, scanning jointly over the finite template family aligns with the likelihood-ratio
structure and is adopted in the stated upper bounds.
Remark 2. For the scan-based procedures in Theorems 1 and 2, the detection thresholds are
governed by the cardinality of the collection B of candidate block locations. In the variance-
shift setting, where the scan is taken jointly over block locations and templates, the thresholds
10

## Page 11

additionally depend on the size m of the template family. In the non-consecutive setting, Bk,n
defined in (1) satisfies |Bk,n| =
 n
k
2 ≤(en/k)2k. Substituting this bound yields thresholds
of order k log(en/k) for the mean-shift scan and k log(en/k) + log m for the variance-shift
scan. In the consecutive setting, both the standard placement family Bcon
k,n and the circular
placement family B◦
k,n satisfy |B| = Θ(n2), yielding logarithmic thresholds of order log n for
the mean-shift scan and log n + log m for the variance-shift scan.
From a computational perspective, scanning over Bk,n requires enumerating
 n
k
2 block lo-
cations and is computationally infeasible in general. In contrast, scans over Bcon
k,n and B◦
k,n ad-
mit polynomial-time implementations via sliding-window or circular convolution techniques.
In the variance-shift setting, the finite-template scan incurs an additional factor m in the
running time due to the enumeration over templates.
3.2
Lower bounds
We now present our information-theoretic lower bounds. Recall that the minimal (or, opti-
mal) achievable risk for testing PH0 versus PH1 satisfies (see, e.g., [Tsy04])
inf
A:Rn×n→{0,1} {PH0(A(X) = 1) + PH1(A(X) = 0)} = 1 −dTV(PH0, PH1),
(35)
and so detection is information-theoretically impossible whenever dTV(PH0, PH1) = o(1). Our
lower bounds depend on the following quantity
Θ⋆≜max
ℓ∈[m]
1
m2k2 log

1
k2
X
u∈[k]×[k]
exp
 m2k2χ2(Pℓ,u∥Q)


.
(36)
The following result gives conditions under which dTV(PH0, PH1) = o(1), thus precluding
successful recovery.
Theorem 3 (Information-theoretic lower bounds). Consider the finite-template submatrix
detection model introduced in Section 2. Let δ = δn > 0 be any sequence such that δn →0,
as n →∞. The following statements hold.
1. Consider the non-consecutive placement family Kk,m,n. If
Θ⋆≤min
1
k, n2 log(1 + δ)
2m2k4

,
(37)
then dTV(PH0, PH1) = o(1), and detection is information-theoretically impossible.
2. Consider the circular consecutive placement family K◦
k,m,n. Assume k ≤n
2. If
Θ⋆≤1
k2 log

1 + n2 log(1 + δ)
4k2m2

,
(38)
then dTV(PH0, PH1) = o(1), and detection is information-theoretically impossible.
11

## Page 12

Corollary 4 (Impossibility for standard consecutive placements). Consider the detection
model in Section 2, under the standard consecutive placement family Kcon
k,m,n. Assume mk =
o(n). If (38) holds, then dTV(PH0, PH1) = o(1), and detection is information-theoretically
impossible.
Proof sketch of Corollary 4. The circular and standard consecutive placement models differ
only through boundary effects. Under the circular model, block locations are translation
invariant, whereas under the standard model only n −k + 1 starting positions are allowed.
For a uniformly random block under the circular model, the probability of wrapping around
the boundary is O(k/n). By a union bound over the m planted blocks, the probability that at
least one block wraps around is O(mk/n). Hence, dTV(P◦
H1, Pcon
H1 ) = O(mk/n). In particular,
if mk = o(n) then dTV(P◦
H1, Pcon
H1 ) = o(1), and the claim follows from Theorem 3.
To build intuition, we discuss the first few steps of the proof of Theorem 3. We begin
with the standard inequality d2
TV(P, Q) ≤
1
2χ2(P∥Q), see, e.g., [Tsy04, Sec. 2]. Thus, to
obtain an impossibility result, it suffices to show that χ2(PH1∥PH0) →0. Due to the product
structure under the null hypothesis and the finite-template construction under the alterna-
tive hypothesis, this task reduces to understanding how the entrywise chi-square distances
χ2(Pℓ,u∥Q) accumulate in the second-moment calculation through overlaps between indepen-
dently drawn planted configurations. This accumulation is, roughly speaking, captured by
Θ⋆. Indeed, for each template ℓ∈[m], (36) aggregates the entrywise divergences χ2(Pℓ,u∥Q)
across template coordinates, and the maximum over ℓcorresponds to the template that
yields the largest contribution to the second moment. As it turns out, Θ⋆characterizes the
exponential growth rate of the second moment of the likelihood ratio and thus determines
the information-theoretic detectability regime.
3.3
Smooth-signal regime
The lower and upper bounds in the previous subsections are general and hold for any set
of templates. In this subsection, we show that for a non-trivial set of structured templates,
these bounds align up to logarithmic factors in a specific regime.
Specifically, for each
template ℓ∈[m], let ϑℓ= (ϑℓ,u)u∈[k]×[k] denote a k × k array of local signal parameters. In
the mean-shift model ϑℓ,u = (Mℓ)u, while in the variance-shift model ϑℓ,u = (Σℓ)u. Define
the signal energy
Eℓ≜
X
u∈[k]×[k]
ϑ2
ℓ,u,
E ≜max
ℓ∈[m] Eℓ.
(39)
Definition 1 (Smooth-signal regime). We say that the finite-template model operates in the
smooth-signal regime if, for all ℓ∈[m] and u ∈[k] × [k]:
(i) Uniform boundedness: supℓ∈[m] ∥ϑℓ∥∞= O(1).
(ii) Non-spikiness: supℓ∈[m]
k2∥ϑℓ∥2
∞
Eℓ
= O(1).
For this non-trivial family of templates, our upper bounds in Theorems 1–2 simplify as
follows.
12

## Page 13

Corollary 5 (Smooth-signal upper bounds). Assume that the smooth-signal regime in Def-
inition 1 holds.
(i) If
E = ω
 n2
m2k2

,
(40)
then the global sum test Asum(X) in (15) satisfies R(Asum) = o(1) under the non-
consecutive placement regime and under both consecutive placement regimes.
(ii) If
E = ω(log |B|),
(41)
then the template-aware scan test Aµ
scan,max(X; B) in (18) satisfies R
 Aµ
scan,max

= o(1).
In particular, for B = Bk,n it suffices that E = ω
 k log n
k

, and for B ∈{Bcon
k,n, B◦
k,n} it
suffices that E = ω(log n).
(iii) If
E = ω
 n2
m2k2

,
(42)
then the quadratic test Aquad(X) in (25) satisfies R(Aquad) = o(1) under the non-
consecutive placement regime and under both consecutive placement regimes.
(iv) If
E = ω(log |B| + log m),
(43)
then the finite-template scan test Aσ
scan(X; B) in (28) satisfies R (Aσ
scan) = o(1).
In
particular, for B = Bk,n it suffices that E = ω
 k log n
k + log m

and for B ∈{Bcon
k,n, B◦
k,n}
it suffices that E = ω(log n + log m).
Note that Corollary 5 implies that the global test outperforms the scan test whenever
log |B| = o

n2
m2k2

, and this is independent of whether the placement is consecutive or not.
Next, our lower bounds in Theorems 3 simplify as follows.
Corollary 6 (Smooth-signal lower bounds). Consider the smooth-signal regime in Defini-
tion 1.
1. Under the non-consecutive placement family Kk,m,n, if
E = o

k ∧
n2
m2k2

,
(44)
then dTV(PH0, PH1) = o(1).
13

## Page 14

Lower bound
Scan (mean)
Scan (variance)
Global tests
Non-consecutive
k ∧
n2
m2k2
k log(n/k)
k log(n/k) + log m
n2
m2k2
Consecutive
log

1 +
n2
k2m2

log n
log n + log m
n2
m2k2
Table 1: Energy scales in the smooth-signal regime in Definition 1. The “Lower bound” column
gives an impossibility condition: if E = o(·) then dTV(PH0, PH1) = o(1).
The scan and global
columns give sufficient conditions: if E = ω(·) then the corresponding test attains vanishing risk.
2. Under the circular consecutive placement family K◦
k,m,n, if
E = o

log

1 +
n2
k2m2

,
(45)
then dTV(PH0, PH1) = o(1).
Corollary 7 (Smooth-signal consecutive placements). Consider the smooth-signal regime in
Definition 1. Under the standard consecutive placement family Kcon
k,m,n. If (45) and mk =
o(n) hold, then dTV(PH0, PH1) = o(1).
Proof of Corollary 7. By Corollary 6, condition (45) implies dTV(PH0, P◦
H1) = o(1) for the
circular consecutive model. If mk = o(n), Corollary 4 gives dTV(P◦
H1, Pcon
H1 ) = o(1). The
result follows from the triangle inequality.
Table 1 summarizes the resulting energy scales and the resulting bounds as captured
by Corollaries 5–7.
It is evident that the lower and upper bounds coincide up to poly-
logarithmic factors. Finally, we note that the classical homogeneous mean-shift submatrix
detection problem [DHB24] is a special case of this framework. Indeed, for homogeneous
templates Mℓ= λIk×k, the signal energy satisfies E = ∥Mℓ∥2
F = k2λ2, and µdet = λmk2.
In this setting, Corollary 6 yields the classical impossibility condition |λ| = o
√
k ∧
n
mk2

,
while the first two items of Theorem 1 show that detection is possible (using the global
and scan detection algorithms) once |λ| = ω
q
log(n/k)
k
∨
n
mk2

. These results coincide with
[DHB24].
4
Proofs
4.1
Proof of Theorem 1
4.1.1
Sum test
Proof. We analyze the Type-I and Type-II errors for the sum statistic (14) and the test (15).
Under H0, the entries of X are i.i.d. N(0, 1), so
Tsum(X) = sign(µdet)
X
i,j∈[n]
Xij ∼N(0, n2).
(46)
14

## Page 15

Hence for any τ ≥0 ,
PH0(Asum(X) = 1) = PH0(Tsum(X) ≥τ)
(47)
= P(N(0, n2) ≥τ)
(48)
≤exp

−τ 2
2n2

.
(49)
Under H1, the law of X is the mixture over the random planted block collection K and,
conditional on K, the random labeling β : K →[m]. Fix any realization (K, β). For each
planted block B ∈K and each (i, j) ∈B, we have
E [Xij|K, β] =
 Mβ(B)

φB(i,j) .
(50)
Since each entry of Mβ(B) appears exactly once over the block B, it holds that
X
(i,j)∈B
E [Xij|K, β] =
X
u,v∈[k]
 Mβ(B)

uv
(51)
Summing over B ∈K and using that β : K →[m] is a bijection,
X
B∈K
X
(i,j)∈B
E [Xij|K, β] =
m
X
ℓ=1
X
u,v∈[k]
(Mℓ)uv = µdet.
(52)
Therefore,
Tsum(X)|(K, β) ∼N
 |µdet| , n2
,
(53)
and hence Tsum(X) ∼N(|µdet| , n2) under H1. Therefore,
PH1(Asum(X) = 0) = PH1(Tsum(X) ≤τ)
(54)
= P
 N
 |µdet| , n2
≤τ

(55)
≤exp

−(τ −|µdet|)2
2n2

.
(56)
for any τ ≤|µdet|. Choosing τ = τsum = |µdet| /2 as in (19), yields R(Asum) ≤2 exp
n
−|µdet|2
8n2
o
.
In particular, if |µdet| /n →∞, then R(Asum) = o(1).
The argument does not depend on the placement family and therefore applies to the
non-consecutive, standard consecutive, and circular consecutive models.
4.1.2
Scan test
Proof. Recall the scan statistic in (16) and the template-aware scan test in (18). In general,
Tµ
scan(X; B, M) = max
B∈B
X
(i,j)∈B
MφB(i,j)Xij,
Aµ
scan,max(X) = 1{Tµ
scan(X; B, M) ≥τ},
(57)
15

## Page 16

where B = Bk,n or B ∈{Bcon
k,n, B◦
k,n}. Let Mmax = Mℓmax where ℓmax ∈arg maxℓ∈[m] ∥Mℓ∥2
F,
Under H0, the entries of X are i.i.d. N(0, 1). Hence for any fixed B ∈B
X
(i,j)∈B
(Mmax)φB(i,j) Xij ∼N(0, ∥Mmax∥2
F),
(58)
since φB maps the indices of B to [k] × [k]. Therefore, applying the union bound over B ∈B
and a Gaussian tail bound yields
PH0
 Aµ
scan,max(X) = 1

= PH0 (Tµ
scan(X; B, Mmax) ≥τ)
(59)
≤
X
B∈B
P
 N(0, ∥Mmax∥2
F) ≥τ

(60)
≤|B| exp
(
−
τ 2
2 ∥Mmax∥2
F
)
.
(61)
Under H1, the law of X is the mixture over the random planted block collection K⋆and,
conditional on K⋆, the random labeling β : K⋆→[m]. Fix any realization (K⋆, β). Define the
planted block carrying Mmax by B⋆
max ≜β−1(ℓmax) which is well-defined and unique since β
is a bijection.
By definition of the scan statistic as a maximum over B, we have
Tµ
scan(X; B, Mmax) ≥
X
(i,j)∈B⋆max
(Mmax)φB⋆max(i,j)Xij.
(62)
Conditionally on (K⋆, β), entries (i, j) on B⋆
max are independent and satisfy
Xij = (Mmax)φB⋆max(i,j) + Zij,
where
Zij
i.i.d.
∼N(0, 1).
(63)
Therefore,
X
(i,j)∈B⋆max
(Mmax)φB⋆max(i,j)Xij =
X
(i,j)∈B⋆max
(Mmax)2
φB⋆max(i,j) +
X
(i,j)∈B⋆max
(Mmax)φB⋆max(i,j)Zij
(64)
= ∥Mmax∥2
F + N
 0, ∥Mmax∥2
F

(65)
∼N
 ∥Mmax∥2
F , ∥Mmax∥2
F

.
(66)
It follows that, conditional on (K⋆, β), for τ ≤|Mmax|2
F a Gaussian lower-tail bound gives
PH1
 Aµ
scan,max(X) = 0 | K⋆, β

= PH1 (Tµ
scan(X; B, Mmax) ≤τ|K⋆, β)
(67)
≤P
 N
 ∥Mmax∥2
F , ∥Mmax∥2
F

≤τ

(68)
≤exp
(
−
 τ −∥Mmax∥2
F
2
2 ∥Mmax∥2
F
)
.
(69)
Since the bound in (69) does not depend on (K⋆, β), it also holds for the marginal Type-II
error under H1, that is
PH1
 Aµ
scan,max(X) = 0

≤exp
(
−
 τ −∥Mmax∥2
F
2
2 ∥Mmax∥2
F
)
.
(70)
16

## Page 17

Combining (66) and (70) yields
R
 Aµ
scan,max(X)

≤|B| exp
(
−
τ 2
2 ∥Mmax∥2
F
)
+ exp
(
−
 τ −∥Mmax∥2
F
2
2 ∥Mmax∥2
F
)
.
(71)
Finally, note that |Bk,n| =
 n
k
2 ≤
  en
k
2k and
Bcon
k,n
 = (n −k + 1)2 ≤n2, with the same
bound holding for B◦
k,n. Substituting the corresponding bound on |B| and choosing the τ as
in (19) yield R
 Aµ
scan,max(X)

= o(1) under the conditions stated in items 2-3 of Theorem
1. The argument depends on B only through its cardinality and therefore applies to the
non-consecutive, standard consecutive, and circular consecutive placement models.
4.2
Proof of Theorem 2
4.2.1
Quadratic test
Proof. Recall the global centered quadratic statistic Tquad in (23), and the quadratic test
Aquad in (25) with the threshold τquad = νdet/2 as in (29). We bound the Type-I and Type-II
error probabilities.
Under H0, Xij ∼N(0, 1), hence EH0

X2
ij −1

= 0 and VarH0(X2
ij −1) = 2. Therefore,
VarH0(Tquad) = 2n2. By Chebyshev’s inequality,
PH0 (Aquad(X) = 1) = PH0 (Tquad(X) ≥τquad)
(72)
≤VarH0 (Tquad(X))
τ 2
quad
=
2n2
(νdet/2)2 = 8n2
ν2
det
= o(1),
(73)
whenever νdet = ω(n).
Under
H1,
conditionally
on
(K, β),
each
planted
entry
satisfies
Xij
∼
N
 0, 1 + (Σβ(B))φB(i,j)

for (i, j) ∈B and Xij ∼N(0, 1) otherwise. Therefore,
EH1

X2
ij −1|K, β

=
(
(Σβ(B))φB(i,j),
(i, j) ∈B ∈K,
0,
otherwise,
(74)
and summing over all entries yields
EH1 [Tquad(X)|K, β] =
X
B∈K
X
(i,j)∈B
(Σβ(B))φB(i,j) = νdet.
(75)
Moreover, since the entries are independent conditional on (K, β) and for Y ∼N(0, σ2) we
have Var(Y 2 −1) = 2σ4,
VarH1 (Tquad(X)|K, β) =
X
i,j∈[n]
VarH1
 X2
ij −1|K, β

(76)
=
X
(i,j)/∈S
B∈K B
Var
 X2
ij −1

+
X
B∈K
X
(i,j)∈B
VarH1
 X2
ij −1|K, β

(77)
17

## Page 18

= 2(n2 −mk2) +
X
B∈K
X
(i,j)∈B
2Var (Xij|K, β)2
(78)
= 2(n2 −mk2) + 2
X
B∈K
X
(i,j)∈B
 1 + (Σβ(B))φB(i,j)
2
(79)
= 2(n2 −mk2) + 2
m
X
ℓ=1
X
u,v∈[k]
(1 + (Σℓ)uv)2
(80)
= 2n2 + 4νdet + 2
m
X
ℓ=1
X
u,v∈[k]
(Σℓ)2
uv.
(81)
Since each template appears exactly once, both EH1 [Tquad(X)|K, β] and VarH1 (Tquad(X)|K, β)
are deterministic (they depend only on the template family), hence they equal the corre-
sponding unconditional quantities.
Applying Chebyshev’s inequality with τquad = νdet/2
gives
PH1 (Aquad(X) = 0) = PH1 (Tquad(X) ≤τquad)
(82)
≤
VarH1 (Tquad(X))
(EH1[Tquad(X)] −τquad)2
(83)
=
2n2 + 4νdet + 2 Pm
ℓ=1
P
u,v∈[k](Σℓ)2
uv
(νdet/2)2
.
(84)
Using maxℓ,u,v(Σℓ)uv ≤ϑ0 we have
m
X
ℓ=1
X
u,v∈[k]
(Σℓ)2
uv ≤ϑ0 νdet,
(85)
The numerator is O(n2+νdet), so the bound is o(1) whenever νdet = ω(n). The argument does
not depend on the placement family and therefore applies to the non-consecutive, standard
consecutive, and circular consecutive models.
4.2.2
Scan test
Proof. Recall the scan statistic in (26) and the finite-template scan test Aσ
scan(X; B) in (28),
where B = Bk,n for non-consecutive placements, B = Bcon
k,n for standard consecutive place-
ments, and B = B◦
k,n for circular consecutive placements.
For any k × k block B ∈B, we define the following distributions:
• Null distribution. We define P(B)
0
≜⊗(i,j)∈BN(0, 1) representing the joint distribution
of the entries in B under the null hypothesis.
• Local alternative. For a given template Σℓ∈S, we define P(B)
ℓ
≜⊗(i,j)∈BN(0, 1 +
(Σℓ)φB(i,j)). This distribution is not the true law under H1; rather, it represents a local
alternative obtained by embedding the template Σℓinto the block B.
18

## Page 19

Accordingly, we define the ℓ-template-matched loglikelihood score of B
Lℓ(B) ≜log P(B)
ℓ
P(B)
0
(X) =
X
(i,j)∈B
1
2

(Σℓ)φB(i,j)
1 + (Σℓ)φB(i,j)
X2
ij −log
 1 + (Σℓ)φB(i,j)

,
(86)
and the corresponding blockwise KL-divergence
KLℓ(B) ≜dKL(P(B)
ℓ
∥P(B)
0 ) = 1
2
X
u,v∈[k]
((Σℓ)uv −log(1 + (Σℓ)uv)) ,
(87)
which does not depend on B. The finite-template scan statistic is
Tσ
scan(X; B) ≜max
ℓ∈[m] max
B∈B Lℓ(B).
(88)
We analyze the Type-I and Type-II error probabilities. Under H0, the entries Xij are i.i.d.
N(0, 1). For any fixed (ℓ, B), Lℓ(B) is a log-likelihood ratio, hence EH0

eLℓ(B)
= 1. By
Markov’s inequality,
PH0 (Lℓ(B) ≥τ σ
scan) ≤e−τ σ
scan.
(89)
Applying a union bound over all ℓ∈[m] and B ∈B,
PH0 (Tσ
scan(X; B) ≥τ σ
scan) ≤
m
X
ℓ=1
X
B∈B
PH0 (Lℓ(B) ≥τ σ
scan) ≤m |B| e−τ σ
scan
(90)
Setting τ σ
scan = (1 + δ)(log |B| + log m) yields PH0 (Tσ
scan(X; B) ≥τ σ
scan) ≤(m|B|)−δ = o(1).
Under H1, fix a realization (K⋆, β). Let ℓ⋆∈arg maxℓ∈[m] KL(Σℓ), where KL(Σ) is defined
in (27), and define B⋆≜β−1(ℓ⋆). Since the scan ranges over all templates and all blocks,
Tσ
scan(X; B) ≥Lℓ⋆(B⋆).
(91)
Conditionally on (K⋆, β), the entries on B⋆satisfy
Xij =
q
1 + (Σℓ⋆)φB⋆(i,j) Zij,
Zij ∼N(0, 1).
(92)
Let (Σℓ⋆)uv ≜σuv. Then, substituting into the definition of Lℓ⋆(B⋆) yields
Lℓ⋆(B⋆) =
X
(i,j)∈B⋆
1
2
 (Σℓ⋆)φB⋆(i,j)Z2
ij −log
 1 + (Σℓ⋆)φB⋆(i,j)

(93)
=
X
u,v∈[k]
1
2
 σuvZ2
uv −log (1 + σuv)

(94)
= 1
2
X
u,v∈[k]
(σuv −log (1 + σuv)) + 1
2
X
(u,v)∈[k]
σuv
 Z2
uv −1

,
(95)
= KL(Σℓ⋆) + 1
2
X
(u,v)∈[k]
σuv
 Z2
uv −1

.
(96)
19

## Page 20

The distribution of Lℓ⋆(B⋆) depends only on the template Σℓ⋆and not on the remainder
of (K⋆, β); hence the conditional and unconditional probabilities coincide. Therefore,
PH1 (Aσ
scan(X; B) = 0) = PH1(Tσ
scan(X; B) ≤τ σ
scan)
(97)
≤PH1 [Lℓ⋆(B⋆) < τ σ
scan]
(98)
= PH1

X
u,v∈[k]
σuv(Z2
uv −1) < 2 (τ σ
scan −KL(Σmax))

.
(99)
Let ∆≜τ σ
scan −KL(Σmax). For any λ > 0, applying Chernoff’s bound gives
PH1 (Aσ
scan(X; B) = 0) ≤e−2λ∆E
h
e−λ P
u,v∈[k] σuv(Z2
uv−1)i
(100)
= e−2λ∆Y
u,v∈[k]
E
h
e−λσuv(Z2
uv−1)i
(101)
= exp


−2λ∆+ λ2 X
u,v∈[k]
σ2
uv



(102)
= exp

−2λ∆+ λ2 ∥Σℓ⋆∥2
F
	
,
(103)
where we use the fact that {Zuv}u,v∈[k] are i.i.d. standard Gaussian with moment generating
function E
h
etZ2i
= (1−2t)−1/2. For λ > 0 satisfying 2λσuv < 1, we obtain E
h
e−λσuv(Z2
uv−1)i
=
eλσuv(1 + 2λσuv)−1/2. Taking log yields
log E
h
e−λσuv(Z2
uv−1)i
= λσuv −1
2 log(1 + 2λσuv) ≤λ2σ2
uv,
(104)
where we used log(1 + x) ≥x −x2
2 for all x ≥0. Finally, E
h
e−λσuv(Z2
uv−1)i
≤exp {λ2σ2
uv}.
Since 0 ≤σuv ≤ϑ0 < 1, the admissibility condition 2λσuv < 1 holds uniformly whenever
λ < 1/(2ϑ0).
We now minimize the right-hand side of (103) over λ. Completing the square,
−2λ∆+ λ2 ∥Σℓ⋆∥2
F = ∥Σℓ⋆∥2
F
 
λ −
∆
∥Σℓ⋆∥2
F
!2
−
∆2
∥Σℓ⋆∥2
F
,
(105)
so the minimum is achieved at λ⋆
=
∆
∥Σℓ⋆∥2
F .
Under the corresponding assumption
of Theorem 2 (items 2- 3), we have maxℓKL(Σℓ) ≫log |B| + log m.
Since τ σ
scan =
(1 + δ) (log |B| + log m), it follows that ∆= DKL(Σℓ⋆) −τ σ
scan →+∞and in particular
0 < ∆< KL(Σℓ⋆) for all sufficiently large n. The minimizer λ⋆=
∆
∥Σℓ⋆∥2
F therefore satisfies
λ⋆> 0. Moreover, since log(1 + x) ≥x −x2
2 for x ≥0, we have for all Σ ∈S
KL(Σ) = 1
2
X
u,v∈[k]
(σuv −log(1 + σuv)) ≤1
4
X
u,v∈[k]
σ2
uv = 1
4 ∥Σ∥2
F ,
(106)
20

## Page 21

hence λ⋆≤KL(Σℓ⋆)/ ∥Σℓ⋆∥2
F ≤1/4. Because 0 ≤σuv ≤ϑ0 < 1, we have 1/(2ϑ0) > 1/2, and
therefore 1/4 < 1/(2ϑ0). Thus λ⋆< 1/(2ϑ0), and the chosen optimizer lies within the range
where the moment generating function bound is valid.
Substituting λ⋆yields
PH1 (Aσ
scan(X; B) = 0) ≤exp
(
−(KL(Σℓ⋆) −τ σ
scan)2
∥Σℓ⋆∥2
F
)
.
(107)
Choosing τ σ
scan = (1 + δ) (log |B| + log m) implies PH1 (Aσ
scan(X; B) = 0) = o(1) whenever
KL (Σℓ⋆) = maxℓ∈[m] KL(Σℓ) ≫log |B| + log m. Hence, under this condition, the Type-II
error probability vanishes.
The argument applies uniformly to all placement families; the dependence on the place-
ment model enters only through the cardinality |B|.
For non-consecutive placements,
|Bk,n| =
 n
k
2 ≤(en/k)2k.
For standard consecutive placements, |Bcon
k,n| = (n −k + 1)2,
and for circular consecutive placements, |B◦
k,n| = n2. Substituting these bounds into thresh-
old τscan = (1+δ)(log |B|+log m) and the condition above yields the thresholds and regimes
stated in the main results.
4.3
Proof of Corollary 5
We treat each item separately.
(i) Mean-shift (global sum test).
Recall µdet = Pm
ℓ=1
P
u,v∈[k](Mℓ)uv. For any matrix
A ∈Rk×k, it holds that
P
u,v∈[k] Auv
 ≤P
u,v∈[k] |Auv| ≤k ∥A∥F. Hence,
|µdet| ≤
m
X
ℓ=1

X
u,v∈[k]
(Mℓ)uv

≤k
m
X
ℓ=1
∥Mℓ∥F ≤mk max
ℓ∈[m] ∥Mℓ∥F .
(108)
Since E = maxℓ∈[m] ∥Mℓ∥2
F, we obtain |µdet| ≤mk
√
E. Therefore, if |µdet| = ω(n), then
necessarily
mk
√
E = ω(n),
(109)
which implies (40). The risk statement follows directly from item 1 in Theorem 1.
(ii) Mean-shift (template-awere scan test).
In the mean-shift model, Eℓ= ∥Mℓ∥2
F,
and E = ∥Mmax∥2
F. Thus, the scan condition of Theorem 1 is equivalent to E = ω(log |B|).
For B = Bk,n, log |B| = Θ
 k log n
k

, and for B ∈

Bcon
k,n, B◦
k,n
	
, log |B| = Θ(log n). The risk
bound follows from Theorem 1.
(iii) Variance-shift (global quadratic test).
Define, for each ℓ∈[m], Sℓ
=
P
u,v∈[k](Σℓ)uv and Eℓ= ∥Σℓ∥2
F, so that νdet = Pm
ℓ=1 Sℓand E = maxℓEℓ. Since (Σℓ)uv ≥0
for all u, v ∈[k], Cauchy–Schwarz yields, for every ℓ,
Sℓ≤k ∥Σℓ∥F = k
p
Eℓ.
(110)
21

## Page 22

Therefore,
νdet =
m
X
ℓ=1
Sℓ≤mk
√
E.
(111)
If the global quadratic test condition (30) holds, i.e. νdet = ω(n), then mk
√
E = ω(n), and
hence
E = ω
 n2
m2k2

,
(112)
which proves (42). The risk bound R(Aquad) = o(1) follows from Theorem 2.
(iv) Variance-shift (finite-template scan).
Recall that
KL(Σℓ) = 1
2
X
u,v∈[k]
((Σℓ)uv −log(1 + (Σℓ)uv)) .
(113)
Under the bounded-variance assumption, 0 ≤(Σℓ)uv ≤ϑ0 < 1. For 0 ≤x ≤ϑ0, define
f(x) ≜x −log(1 + x). Then f(0) = f ′(0) = 0 and f ′′(x) =
1
(1+x)2 ∈
h
1
(1+ϑ0)2, 1
i
. By Taylor’s
theorem, for each x ∈[0, ϑ0],
1
2(1 + ϑ0)2x2 ≤f(x) ≤1
2x2.
(114)
Applying this bound entrywise yields
1
4(1 + ϑ0)2 ∥Σℓ∥2
F ≤KL(Σℓ) ≤1
4 ∥Σℓ∥2
F .
(115)
Hence,
max
ℓ∈[m] KL(Σℓ) = Θ

max
ℓ∈[m] ∥Σℓ∥2
F

= Θ(E).
(116)
The scan condition in Theorem 2 is therefore equivalent to
E = ω(log |B| + log m),
(117)
and the risk bound follows from Theorem 2.
4.4
Proof of Theorem 3
We use a single second-moment bound that applies to both the non-consecutive and circular
consecutive models, and then plug in the corresponding overlap estimates for each placement
family.
22

## Page 23

Likelihood ratio and second moment.
In order to lower bound the optimal risk, we ap-
ply the second-moment method, which reduces the problem to bounding the second moment
of the likelihood ratio under H0. In particular,
dTV(PH0, PH1) ≤1
2
p
χ2(PH1∥PH0) = 1
2
p
EH0 [Ln(X)2] −1.
(118)
Thus, it suffices to show that EH0 [Ln(X)2] ≤1 + o(1), which implies χ2(PH1∥PH0) = o(1)
and hence dTV(PH0, PH1) = o(1).
Recall that under H0, the entries {Xij}i,j∈[n] are independent with common density Q,
that is PH0 = Q⊗n2. Under H1, we draw a random set K ∈K of m disjoint blocks (either
Kk,m,n, Kcon
k,m,n or K◦
k,m,nas defined in (3)-(5)). Given K, a uniform random bijection β : K →
[m] assigns a label to each block, so that β(B) = ℓindicates that block B carries template ℓ.
Thus,
PH1 = EK,β [PK,β] .
(119)
Conditional on (K, β), the entries remain independent, and for each (i, j) ∈[n]2,
Xij ∼
(
Pβ(B),φB(i,j),
if (i, j) ∈B for some B ∈K,
Q,
otherwise.
(120)
Here, for each block B ∈K, φB : B →[k] × [k] denotes the deterministic coordinate map
defined in (7), and {Pℓ,u}ℓ∈[m], u∈[k]×[k] is the unified family of signal densities introduced in
Section 2. We assume that
Pℓ,u ≪Q,
for all ℓ∈[m], u ∈[k] × [k],
(121)
so that likelihood ratios are well-defined. For each ℓ∈[m] and u ∈[k] × [k], define the
entrywise likelihood ratio
Lℓ,u(x) ≜Pℓ,u(x)
Q(x) ,
so that
EH0 [Lℓ,u(Xij)] = 1.
(122)
Then the conditional likelihood ratio (the Radon–Nikodym derivative of PK,β with respect
to PH0) is
L(X|K, β) =
Y
B∈K
Y
(i,j)∈B
Lβ(B),φB(i,j)(Xij),
(123)
and the mixture likelihood ratio is
Ln(X) ≜PH1
PH0
(X) = EK,β [L(X|K, β)] .
(124)
23

## Page 24

General second moment reduction.
We are now in a position to compute the second
moment. Let (K′, β′) be an independent copy of (K, β). Then, by Fubini’s theorem,
EH0

L2
n(X)

= E(K,β)⊥⊥(K′,β′) [L (X|K, β) L (X|K′, β′)] .
(125)
Since under PH0 the entries {Xij}i,j∈[n] are independent, the inner expectation factorizes
entry-wise. All coordinates (i, j) outside the overlap of the planted unions contribute factor
1, since EH0[Lℓ,u(Xij)] = 1 for all (ℓ, u). Only indices (i, j) that lie in both planted unions
contribute non-trivially. Define the overlap set and its size by
K ∩K′ ≜
 [
B∈K
B
! \  [
B′∈K′
B′
!
=
[
(B,B′)∈K×K′
B ∩B′,
H = |K ∩K′| ,
(126)
Since blocks are disjoint within each planted collection, for each (i, j) ∈K ∩K′ there exist
unique blocks B ∈K and B′ ∈K′ such that (i, j) ∈B ∩B′. Therefore,
EH0

L2
n(X)

= E(K,β)⊥⊥(K′,β′)


Y
(B,B′)∈K×K′
Y
(i,j)∈B∩B′
ρ (β(B), β′(B′); φB (i, j) φB′ (i, j))

, (127)
where
ρ(ℓ, ℓ′; u, u′) ≜EZ∼Q [Lℓ,u(Z) Lℓ′,u′(Z)] .
(128)
Lemma 1 (Cauchy–Schwarz domination of the overlap factor). Assume Pℓ,u ≪Q for all
ℓ∈[m], u ∈[k] × [k], and write χ2
ℓ,u ≜χ2(Pℓ,u∥Q). Then, for all (ℓ, u), (ℓ′, u′),
ρ(ℓ, ℓ′; u, u′) ≤exp
1
2
 χ2
ℓ,u + χ2
ℓ′,u′

.
(129)
Proof. By Cauchy–Schwarz,
ρ(ℓ, ℓ′; u, u′) =
Z
Lℓ,u(x)Lℓ′,u′(x) Q(x)dx
(130)
≤
Z
L2
ℓ,u(x)Q(x)dx
1/2 Z
L2
ℓ′,u′(x)Q(x)dx
1/2
(131)
=
q
1 + χ2
ℓ,u
q
1 + χ2
ℓ′,u′
(132)
≤exp
1
2χ2
ℓ,u + 1
2χ2
ℓ′,u′

,
(133)
using log(1 + x) ≤x for x ≥0.
Applying Lemma 1 and then Cauchy–Schwarz to separate the contributions of β and β′
yields
EH0

L2
n(X)

≤E(K,β)⊥⊥K′

exp



X
(B,B′)∈K×K′
X
(i,j)∈B∩B′
χ2
β(B),φB(i,j)





(134)
24

## Page 25

For each block-pair (B, B′), define
SB,B′(β) ≜
X
(i,j)∈B∩B′
χ2
β(B),φB(i,j).
(135)
Let
F ≜σ (K, {HB,B′ : B ∈K, B′ ∈K′}) ,
HB,B′ ≜|B ∩B′| .
(136)
Conditioning on F and applying conditional H¨older (e.g. [Dur19, Ch. 4]) with exponent
q = m2 gives
Eβ⊥⊥K′|F

exp



X
(B,B′)∈K×K′
SB,B′(β)




F

≤
Y
(B,B′)∈K×K′
 Eβ⊥⊥K′|F

eq SB,B′(β) F
1/q .
(137)
Since β is a uniform bijection K →[m], the random label β(B) of each fixed B ∈K is uniform
on [m], hence for each block-pair (B, B′),
Eβ⊥⊥K′|F

eq SB,B′(β) F

= 1
m
m
X
ℓ=1
EK′|F

exp


q
X
(i,j)∈B∩B′
χ2
ℓ,φB(i,j)




F

.
(138)
For each block-pair (B, B′), we define the random index set
UB,B′ ≜{φB(i, j) : (i, j) ∈B ∩B′} ⊂[k] × [k],
such that |UB,B′| = HB,B′.
(139)
Lemma 2 (Conditional exchangeability of overlap coordinates). Fix a block pair (B, B′) ∈
K×K′ under either the non-consecutive placement model or the circular consecutive placement
model, and let F be defined as in (136). Then, conditional on F, for all u, u′ ∈[k] × [k],
P {u ∈UB,B′|F} = P {u′ ∈UB,B′|F} .
(140)
In particular, for all u ∈[k] × [k],
E [1 {u ∈UB,B′} |F] = HB,B′
k2 .
(141)
Consequently, for any deterministic function f : [k] × [k] →R,
E

X
u∈UB,B′
f(u)

F

= HB,B′
k2
X
u∈[k]×[k]
f(u).
(142)
Proof of Lemma 2. Fix (B, B′) and condition on F, so that K and all overlap sizes HB,B′ are
fixed. We first prove (140).
Non-consecutive model. Conditional on F, the remaining randomness is the placement
of K′ subject to the overlap sizes. Under the non-consecutive placement rule, the conditional
law does not privilege any specific location inside a fixed block B: for any u, u′ ∈[k] × [k],
25

## Page 26

there exists a relabeling of the rows and columns inside B that maps the event {u ∈UB,B′}
to {u′ ∈UB,B′} while leaving F unchanged. Hence the conditional inclusion probabilities are
equal.
Circular consecutive model. In the circular consecutive placement model, the planted
row-interval and column-interval of B′ are generated by uniform starting points on the cor-
responding cycles. Conditional on F, the resulting law of the overlap location inside B is
invariant under simultaneous cyclic shifts of the k template rows and of the k template
columns. Since such shifts act transitively on [k]×[k], the conditional inclusion probabilities
are equal for all u, u′ ∈[k] × [k].
This proves (140). Now,
X
u∈[k]×[k]
1 {u ∈UB,B′} = |UB,B′| = |B ∩B′| = HB,B′.
(143)
Taking conditional expectations and using (140) yields
k2 E [1 {u ∈UB,B′} |F] = HB,B′,
(144)
which implies (141). Finally, by linearity,
E

X
u∈UB,B′
f(u)

F

=
X
u∈[k]×[k]
f(u) E [1 {u ∈UB,B′} |F] = HB,B′
k2
X
u∈[k]×[k]
f(u),
(145)
proving (142).
Fix a block pair (B, B′) and a label ℓ∈[m]. Recall that q = m2 and
SB,B′(ℓ) =
X
u∈UB,B′
χ2
ℓ,u,
HB,B′ = |UB,B′|,
(146)
If HB,B′ = 0 then SB,B′(ℓ) = 0 and both sides below equal 1.
Assume henceforth that
HB,B′ ≥1. Next, we apply Jensen’s inequality for the convex function eqx:
exp {qSB,B′} = exp


q
X
u∈UB,B′
χ2
ℓ,u


≤
1
HB,B′
X
u∈UB,B′
exp

qHB,B′χ2
ℓ,u
	
,
(147)
where we use the convention
1
|U|
P
u∈U g(u) = 1 when U = ∅.
Taking conditional expectation with respect to K′ given F, and using Lemma 2 with
f(u) = exp(qHB,B′χ2
ℓ,u), yields
EK′|F [exp {qSB,B′(ℓ)}] ≤1
k2
X
u∈[k]×[k]
exp

qHB,B′χ2
ℓ,u
	
= Aℓ(HB,B′) ,
(148)
where the deterministic function Aℓ(h) is defined for h ∈[0, k2] as follows
Aℓ(h) ≜1
k2
X
u∈[k]×[k]
eqhχ2
ℓ,u.
(149)
26

## Page 27

Lemma 3 (Log-convex interpolation). For each ℓ∈[m], the function h 7→log Aℓ(h) is
convex on [0, k2] and satisfies Aℓ(0) = 1. In particular, for every h ∈[0, k2]
log Aℓ(h) ≤h
k2 log Aℓ(k2),
equivalently
Aℓ(h) ≤exp {θℓh} ,
(150)
where
θℓ≜1
k2 log Aℓ(k2) = 1
k2 log

1
k2
X
u∈[k]×[k]
exp

qk2χ2
ℓ,u
	

.
(151)
Proof of Lemma 3. Let xℓ,u ≜exp

qχ2
ℓ,u
	
> 0.
Then Aℓ=
1
k2
P
u∈[k]×[k] xh
ℓ,u, and h 7→
log P
u x h
ℓ,u is convex as a log-sum-exp of affine functions h log xℓ,u. Hence h 7→log Aℓ(h) is
convex as well. Since Aℓ(0) = 1, convexity implies
log Aℓ(h) ≤h
k2 log Aℓ(k2) +

1 −h
k2

log Aℓ(0) = h
k2 log Aℓ(k2).
(152)
Combining (148) with Lemma 3 gives, for every h ∈[0, k2],
EK′|F [exp {qSB,B′(ℓ)} |F] ≤Aℓ(HB,B′) ≤exp {θℓHB,B′} .
(153)
Recall the definition of the effective χ2 energy in (36)
Θ⋆≜max
ℓ∈[m]
θℓ
q = max
ℓ∈[m]
1
qk2 log

1
k2
X
u∈[k]×[k]
exp

qk2χ2
ℓ,u
	

,
q = m2.
(154)
Substituting (153) into (138) yields
Eβ⊥⊥K′|F

eqSB,B′(β) F

= 1
m
m
X
ℓ=1
EK′|F [exp {qSB,B′(ℓ)} |F]
≤1
m
m
X
ℓ=1
exp {θℓHB,B′} ≤exp {qΘ⋆HB,B′} .
(155)
Plugging this bound into (137) and then taking expectation over F gives
EH0

L2
n(X)

≤EF
"
exp
(
Θ⋆X
B∈K
X
B′∈K′
|B ∩B′|
)#
.
(156)
By construction, the exponential term in (156) is F-measurable.
27

## Page 28

4.4.1
Non-consecutive placements
Recall that under the non-consecutive placements model, each block is constructed as B =
S × T. For each block-pair (B, B′) ∈K × K′, each constructed as B = S × T, B′ = S′ × T′, we
define their row and column overlap sizes
RB,B′ ≜|S ∩S′| ,
CB,B′ ≜|T ∩T′| ,
such that |B ∩B′| = RB,B′CB,B′.
(157)
For a fixed pair (B, B′) with B = S × T and B′ = S′ × T′, the overlap sizes satisfy
RB,B′ ∼Hypergeometric(n, k, k),
CB,B′ ∼Hypergeometric(n, k, k).
(158)
Since the row and column sets are sampled without replacement, the corresponding coordi-
nates are negatively associated [JDP83]. As x 7→eΘ⋆x is increasing, it follows that
EF

exp


Θ⋆
X
(B,B′)∈K×K′
|B ∩B′|




≤
Y
(B,B′)∈K×K′
E [exp {Θ⋆RB,B′CB,B′}]
(159)
= (E [exp {Θ⋆RC}])m2 ,
(160)
where R, C ∼Hypergeometric(n, k, k). We use that a Hypergeometric(n, k, k) random variable
is stochastically dominated by W ∼Binomial(k, k/n); see, e.g., [Hoe63]. Thus, letting W′ be
an independent copy of W,
E [exp {Θ⋆RC}] = E [exp {Θ⋆WW′}] = E
"
1 + k
n
 eΘ⋆W −1
k#
.
(161)
Assume Θ⋆≤1
k. Then 0 ≤Θ⋆W ≤1, and we use the bound ex −1 ≤x + x2 for 0 ≤x ≤1.
This yields
E [exp {Θ⋆WW′}] ≤E
"
1 + k
n
 Θ⋆W + (Θ⋆)2W2k#
(162)
≤E
"
1 + 2k
nΘ⋆W
k#
(163)
≤E
h
e2 k2
n Θ⋆Wi
(164)
=

1 + k
n

e2 k2
n Θ⋆−1
k
.
(165)
Therefore,
EH0

L2
n(X)

≤

1 + k
n

e2 k2
n Θ⋆−1
km2
.
(166)
This is at most 1 + δ provided
k
n

e2 k2
n Θ⋆−1

≤(1 + δ)
1
km2 −1.
(167)
28

## Page 29

Since (1 + δ)
1
km2 −1 ≥log(1+δ)
km2
this implied by
Θ⋆≤n
2k2 log

1 + n log(1 + δ)
m2k2

.
(168)
Combining this with the condition Θ⋆≤1
k yields, we obtain that the second-moment is at
most 1 + δ if
Θ⋆≤min
1
k, n
2k2 log

1 + n log(1 + δ)
m2k2

(169)
= min
1
k, n2 log(1 + δ)
2m2k4

,
(170)
where the last equality holds when
n2
m2k2 = o(1). This establishes the claimed bound for the
non-consecutive model.
4.4.2
Circular consecutive placements
The key distinction from the non-consecutive case lies in the distribution of the overlap
size |B ∩B′| under the circular consecutive placement family. Recall that B = S × T and
B′ = S′ × T′. For a fixed pair (B, B′), we have
|B ∩B′| = |S ∩S′| · |T ∩T′|.
(171)
Under the circular consecutive model, the row-interval S′ is generated by a uniform starting
point on the cycle of length n (and similarly for T′). Therefore, for k ≤n/2,
P (|S ∩S′| = z) =







n−2k+1
n
,
for z = 0,
2
n,
for z = 1, . . . , k −1,
1
n,
for z = k,
(172)
and the same distribution holds for |T ∩T′|. Let Z and Z′ be independent random variables
with distribution (172). Since x 7→eΘ⋆x is increasing, an application of negative association
yields
EF

exp


Θ⋆
X
(B,B′)∈K×K′
|B ∩B′|




≤
Y
(B,B′)∈K×K′
E [exp {Θ⋆|B ∩B′|}]
(173)
= (E [exp {Θ⋆ZZ′}])m2 .
(174)
Next,
E [exp {Θ⋆ZZ′}] = EZ′
"
n −2k + 1
n
+ 2
n
k−1
X
z=1
eΘ⋆zZ′ + eΘ⋆kZ′
n
#
(175)
≤EZ′
n −2k + 1
n
+ 2(k −1)
n
eΘ⋆kZ′ + eΘ⋆kZ′
n

(176)
29

## Page 30

≤EZ′
n −2k
n
+ 2k
n eΘ⋆kZ′
(177)
= n −2k
n
+ 2k
n E
h
eΘ⋆kZ′i
(178)
= n −2k
n
+ 2k
n
 
n −2k + 1
n
+ 2
n
k−1
X
z=1
eΘ⋆zk + eΘ⋆k2
n
!
(179)
≤n −2k
n
+ 2k
n
n −2k
n
+ 2k
n eΘ⋆k2
(180)
= 1 + 4k2
n2

eΘ⋆k2 −1

.
(181)
Therefore,
(E [exp {Θ⋆ZZ′}])m2 ≤

1 + 4k2
n2

eΘ⋆k2 −1
m2
.
(182)
This is at most 1 + δ if
4k2
n2

eΘ⋆k2 −1

≤(1 + δ)
1
m2 −1.
(183)
Since (1 + δ)
1
m2 −1 ≥log(1+δ)
m2
, it suffices that
Θ⋆≤1
k2 log

1 + n2 log(1 + δ)
4k2m2

.
(184)
4.5
Proof of Corollary 4
Let P◦
H1 denote the mixture alternative under the circular consecutive placement family, and
let Pcon
H1 denote the mixture alternative under the standard consecutive placement family.
The two models differ only in the distribution of the planted support. Under the circular
model, each row and column interval is generated by a uniform starting index in [n], allowing
wrap-around. Under the standard model, only the n −k + 1 non-wrap starting positions are
allowed.
For a one-dimensional circular interval with start i ∼Unif([n]), the interval wraps if and
only if i ∈{n −k + 2, . . . , n}. Hence,
P(wrap) = k −1
n
.
(185)
For a block B = S ×T in the circular model, the row and column starts are independent and
uniform on [n]. Therefore,
P(B wraps) ≤P(S wraps) + P(T wraps) = 2(k −1)
n
,
(186)
30

## Page 31

where B is said to wrap if either its row interval or its column interval wraps around the
boundary. Since the support contains m blocks,
P(∃wrapped block) ≤2m(k −1)
n
.
(187)
We couple the two support distributions as follows. Sample K◦from the circular place-
ment family. If no block wraps, set Kcon = K◦. Otherwise, sample Kcon independently from
the standard placement family. Then
P(Kcon ̸= K◦) ≤2m(k −1)
n
.
(188)
Conditional on a support K, both models draw a uniform bijection β : K →[m] and then
generate X with independent entries according to the same rule on S
B∈K B and according to
Q elsewhere. On the event Kcon = K◦we use the same labeling β and the same draw of X in
both models, and hence
dTV
 P◦
H1, Pcon
H1

≤P(Kcon ̸= K◦) ≤2m(k −1)
n
.
(189)
By Theorem 3, if (38) holds then
dTV
 PH0, P◦
H1

= o(1).
(190)
Using the triangle inequality,
dTV
 PH0, Pcon
H1

≤dTV
 PH0, P◦
H1

+ dTV
 P◦
H1, Pcon
H1

≤o(1) + 2m(k −1)
n
.
(191)
If mk = o(n), the right-hand side is o(1), completing the proof.
4.6
Proof of Corollary 6
We work under the problem formulation of Section 2 and assume the smooth-signal regime
from Definition 1. For each ℓ∈[m] and u ∈[k] × [k], let ϑℓ,u denote the local deviation from
Q (the mean in the mean-shift model and the variance shift in the variance-shift model).
Recall the definition of Θ⋆in (36) and the energy definition in (39)
Eℓ≜
X
u∈[k]×[k]
ϑ2
ℓ,u,
E ≜max
ℓ∈[m] Eℓ.
(192)
By Definition 1, there exist constants ϑ0 > 0 and Csp > 0, independent of n, such that for
all sufficiently large n, for every ℓ∈[m] and u ∈[k] × [k],
|ϑℓ,u| ≤ϑ0,
max
u∈[k]×[k] ϑ2
ℓ,u ≤Csp
Eℓ
k2.
(193)
In the smooth-signal regime, Corollary 6 states the impossibility conditions directly in terms
of E, whereas Theorem 3 is formulated in terms of Θ⋆. We therefore begin by relating Θ⋆to
E.
31

## Page 32

Bound Θ⋆by the energy in the smooth-signal regime.
As in the proof of Theorem 3,
we write χ2
ℓ,u ≜χ2(Pℓ,u∥Q). Fix a constant Cχ ≥0 and assume for the moment that the
following quadratic per-entry bound holds for all ℓ∈[m] and u ∈[k] × [k].
χ2
ℓ;u ≤Cχ ϑ2
ℓ,u.
(194)
Under this assumption, we show that in the smooth-signal regime,
Θ⋆≤CχCsp
E
k2.
(195)
Fix ℓ∈[m] and write χ2
max(ℓ) ≜maxu∈[k]×[k] χ2
ℓ;u.
Since exp(·) is increasing, for every
u ∈[k] × [k] we have
exp

m2k2χ2
ℓ,u
	
≤exp

m2k2χ2
max(ℓ)
	
,
(196)
hence
1
k2
X
u∈[k]×[k]
exp

m2k2χ2
ℓ,u
	
≤exp

m2k2χ2
max(ℓ)
	
.
(197)
Taking log and dividing by m2k2 gives
1
m2k2 log

1
k2
X
u∈[k]×[k]
exp

m2k2χ2
ℓ,u
	

≤χ2
max(ℓ).
(198)
Maximizing over ℓyields the clean bound
Θ⋆≤max
ℓ∈[m] max
u∈[k]×[k] χ2
ℓ,u.
(199)
Now apply the assumed per-entry bound (194)
Θ⋆≤Cχ max
ℓ∈[m] max
u∈[k]×[k] ϑ2
ℓ,u.
(200)
Finally,
use
the
non-spikiness
condition
in
the
smooth-signal
regime
(ii),
i.e.,
maxu∈[k]×[k] ϑ2
ℓ,u ≤Csp
Eℓ
k2 for all ℓ∈[m], which gives
Θ⋆≤CχCsp max
ℓ
Eℓ
k2 = CχCsp
E
k2.
(201)
This establishes (195) assuming (194).
Conclude Corollary 6 from Theorem 3.
We treat the non-consecutive and circular
consecutive cases separately.
32

## Page 33

1. Non-consecutive placements. Assume E = o

k ∧
n2
m2k2

. Then by (195),
Θ⋆≤CχCsp
E
k2 = o
1
k ∧
n2
m2k4

.
(202)
In particular, since Cχ and Csp are fixed constants and log(1 + δ) > 0 is fixed, the
above o(·) bound implies that for all sufficiently large n,
Θ⋆≤min
1
k, n2 log(1 + δ)
2m2k4

,
(203)
so the non-consecutive impossibility condition in Theorem 3 holds, and therefore
dTV(PH0, PH1) = o(1).
2. Circular consecutive placements. Assume E = o

log

1 +
n2
k2m2

. Then (195) gives
Θ⋆≤CχCsp
E
k2 = o
 1
k2 log

1 +
n2
k2m2

,
(204)
and hence, for all large n,
Θ⋆≤1
k2 log

1 + n2 log(1 + δ)
4k2m2

.
(205)
Thus the circular consecutive impossibility condition in Theorem 3 holds, and therefore
dTV(PH0, PH1) = o(1).
Verify the per-entry quadratic bound (194) in the two Gaussian models.
It re-
mains to justify the temporary assumption (194).
1. Mean-shift. Here Q = N(0, 1) and Pℓ;u = N((Mℓ)u, 1), so
χ2
ℓ,u = exp

(Mℓ)2
u
	
−1.
(206)
Under the uniform boundedness condition (i) |(Mℓ)u| ≤ϑ0, convexity of ex on [0, ϑ2
0]
gives
e(Mℓ)2
u −1 ≤eϑ2
0 −1
ϑ2
0
(Mℓ)2
u,
(207)
so (194) holds with Cχ = eϑ2
0−1
ϑ2
0
and ϑℓ,u = (Mℓ)u.
2. Variance-shift. Here Q = N(0, 1) and Pℓ;u = N ((0, 1 + (Σℓ)u), with 0 ≤(Σℓ)u ≤ϑ0 <
1. A direct calculation gives
χ2
ℓ;u =
1
p
1 −(Σℓ)2
u
−1.
(208)
33

## Page 34

Let g(t) = (1−t)−1/2 −1 on [0, 1). Then g(0) = 0 and g′(t) = 1
2(1−t)−3/2 is increasing,
hence for t ∈[0, ϑ2
0],
g(t) ≤
 
sup
s∈[0,ϑ2
0]
g′(s)
!
t =
1
2(1 −ϑ2
0)3/2t.
(209)
Taking t = (Σℓ)2
u yields
χ2
ℓ;u ≤
1
2(1 −ϑ2
0)3/2 (Σℓ)2
u,
(210)
so (194) holds with Cχ =
1
2(1−ϑ2
0)3/2 and ϑℓ,u = (Σℓ)u.
Combining these three steps completes the proof.
5
Conclusion and Future Directions
We analyzed detection in a finite-template inhomogeneous submatrix model for Gaussian
matrices. The model permits multiple planted submatrices with coordinate-dependent sig-
nal structure, interpolating between the classical homogeneous mean-shift setting and fully
heterogeneous alternatives.
We established information-theoretic lower bounds via a χ2
second-moment argument and matching upper bounds based on global and scan statistics.
In the smooth-signal regime, and under the associated sparsity conditions on (m, k, n), the
upper and lower bounds coincide up to logarithmic factors for both non-consecutive and
consecutive placement models.
Our results highlight several structural features of submatrix detection.
In the non-
consecutive model, the number of planted submatrices affects the statistical boundary, and
the information-theoretic threshold can lie strictly below the scan-based detection threshold
in a nontrivial parameter regime. In the consecutive model, the scan procedure attains the
information-theoretic threshold up to logarithmic factors. Under mild regularity conditions,
heterogeneous templates exhibit the same detection scaling as the homogeneous case when
expressed in terms of the total signal energy. The classical homogeneous model appears as
a special case within our framework.
Several directions remain open.
In the non-consecutive setting, as can be seen from
Theorems 1–2, both the computationally efficient global test and the computationally
expensive scan test are needed to characterize the statistical limit.
This suggests a
statistical–computational gap: there is a parameter regime in which detection is information-
theoretically possible (e.g., via the scan test), but no efficient algorithm is currently known.
It would be interesting to provide evidence for such a gap using, for example, the frame-
work of low-degree polynomials (see, e.g., [HB18, KWB22]). On the modeling side, while
the finite-template assumption isolates structured heterogeneity, extending the analysis to
fully heterogeneous signals without template constraints would require new tools to control
the overlap structure and second moments. It is also natural to consider non-Gaussian noise
models or alternatives arising from more general exponential families, and to ask whether the
effective energy characterization persists in those settings. More broadly, the finite-template
34

## Page 35

model offers a controlled setting in which to study how structured inhomogeneity influences
statistical and computational limits in high-dimensional detection problems. Similar phe-
nomena are likely to arise in related structured matrix models. Finally, although this paper
focuses on the detection problem, the recovery variant (i.e., identifying the planted submatrix
exactly or partially) is also of interest.
References
[ACCD10] Ery Arias-Castro, Emmanuel Cand`es, and Arnaud Durand.
Detection of an
anomalous cluster in a network. Annals of Statistics, 39, Jan. 2010.
[ACV14]
Ery Arias-Castro and Nicolas Verzelen. Community detection in dense random
networks. The Annals of Statistics, 42(3):940–969, 2014.
[BBH18]
Matthew Brennan, Guy Bresler, and Wasim Huleihel. Reducibility and compu-
tational lower bounds for problems with planted sparse structure. In Proceedings
of the 31st Conference On Learning Theory, volume 75, pages 48–166, 06–09 Jul
2018.
[BBH19]
Matthew Brennan, Guy Bresler, and Wasim Huleihel. Universality of computa-
tional lower bounds for submatrix detection. In Proceedings of the Thirty-Second
Conference on Learning Theory, volume 99, pages 417–468, 25–28 Jun 2019.
[BBS20]
Tamir Bendory, Alberto Bartesaghi, and Amit Singer.
Single-particle cryo-
electron microscopy: Mathematical theory, computational challenges, and op-
portunities. IEEE signal processing magazine, 37(2):58–76, 2020.
[BDN17]
Shankar Bhamidi, Partha Dey, and Andrew Nobel. Energy landscape for large
average submatrix detection problems in gaussian random matrices. Probability
Theory and Related Fields, 168, 08 2017.
[BI13]
Cristina Butucea and Yuri I Ingster. Detection of a sparse submatrix of a high-
dimensional noisy matrix. Bernoulli, 19(5B):2652–2688, 2013.
[BKR+11] Sivaraman Balakrishnan, Mladen Kolar, Alessandro Rinaldo, Aarti Singh, and
Larry Wasserman. Statistical and computational tradeoffs in biclustering. In
NIPS 2011 workshop on computational trade-offs in statistical learning, volume 4,
2011.
[BMR+19] Tristan Bepler, Andrew Morin, Micah Rapp, Julia Brasch, Lawrence Shapiro,
Alex J Noble, and Bonnie Berger.
Positive-unlabeled convolutional neural
networks for particle picking in cryo-electron micrographs.
Nature methods,
16(11):1153–1160, 2019.
[BMS15]
Xiao-Chen Bai, Greg McMullan, and Sjors HW Scheres. How cryo-EM is revo-
lutionizing structural biology. Trends in biochemical sciences, 40(1):49–57, 2015.
35

## Page 36

[CC18]
Utkan Onur Candogan and Venkat Chandrasekaran. Finding planted subgraphs
with few eigenvalues using the schur–horn relaxation. SIAM Journal on Opti-
mization, 28(1):735–759, 2018.
[CLR17]
Tony Cai, Tengyuan Liang, and Alexander Rakhlin. Computational and statis-
tical boundaries for submatrix localization in a large noisy matrix. Annals of
Statistics, 45(4):1403–1430, 08 2017.
[CX16]
Yudong Chen and Jiaming Xu. Statistical-computational tradeoffs in planted
problems and submatrix localization with a growing number of clusters and sub-
matrices. Journal of Machine Learning Research, 17(27):1–57, 2016.
[DHB24]
Marom Dadon, Wasim Huleihel, and Tamir Bendory. Detection and recovery of
hidden submatrices. IEEE Transactions on Signal and Information Processing
over Networks, 10:69–82, 2024.
[Dur19]
Rick Durrett. Probability: theory and examples, volume 49. Cambridge university
press, 2019.
[EH25]
Dor Elimelech and Wasim Huleihel. Detecting arbitrary planted subgraphs in
random graphs. In Proceedings of Thirty Eighth Conference on Learning Theory,
volume 291, pages 1691–1798. PMLR, 30 Jun–04 Jul 2025. Available at https:
//proceedings.mlr.press/v291/elimelech25a.html.
[ELS20]
Amitay Eldar, Boris Landa, and Yoel Shkolnisky.
KLT picker:
Particle
picking using data-driven optimal templates.
Journal of structural biology,
210(2):107473, 2020.
[EWD+24] Amitay Eldar, Keren Mor Waknin, Samuel Davenport, Tamir Bendory, Armin
Schwartzman, and Yoel Shkolnisky. Object detection under the linear subspace
model with application to cryo-EM images. arXiv preprint arXiv:2405.00364,
2024.
[HAS18]
Ayelet Heimowitz, Joakim And´en, and Amit Singer. APPLE picker: Automatic
particle picking, a low-effort cryo-EM framework. Journal of structural biology,
204(2):215–227, 2018.
[HB18]
Samuel Hopkins B. Statistical Inference and the Sum of Squares Method. PhD
thesis, Cornell University, 2018.
[Hoe63]
Wassily Hoeffding. Probability inequalities for sums of bounded random vari-
ables. Journal of the American statistical association, 58(301):13–30, 1963.
[Hul22]
Wasim Huleihel. Inferring hidden structures in random graphs. IEEE Transac-
tions on Signal and Information Processing over Networks, 8:855–867, 2022.
[HWX15]
Bruce Hajek, Yihong Wu, and Jiaming Xu. Computational lower bounds for
community detection on random graphs. In Proceedings of The 28th Conference
on Learning Theory, volume 40, pages 899–928, 03–06 Jul 2015.
36

## Page 37

[HWX16]
Bruce Hajek, Yihong Wu, and Jiaming Xu. Achieving exact cluster recovery
threshold via semidefinite programming.
IEEE Transactions on Information
Theory, 62(5):2788–2797, 2016.
[HWX17]
Bruce Hajek, Yihong Wu, and Jiaming Xu. Information limits for recovering a
hidden community. IEEE Transactions on Information Theory, 63(8):4729–4745,
2017.
[JDP83]
Kumar Joag-Dev and Frank Proschan. Negative association of random variables
with applications. The Annals of Statistics, pages 286–295, 1983.
[KBRS11] Mladen Kolar, Sivaraman Balakrishnan, Alessandro Rinaldo, and Aarti Singh.
Minimax localization of structural information in large noisy matrices. In Ad-
vances in Neural Information Processing Systems, pages 909–917, 2011.
[KWB22]
Dmitriy Kunisky, Alexander S. Wein, and Afonso S. Bandeira. Notes on compu-
tational hardness of hypothesis testing: Predictions using the low-degree likeli-
hood ratio. In Mathematical Analysis, its Applications and Computation, pages
1–50. Springer International Publishing, 2022.
[Lyu19]
Dmitry Lyumkis. Challenges and opportunities in cryo-EM single-particle anal-
ysis. Journal of Biological Chemistry, 294(13):5181–5197, 2019.
[Mon15]
Andrea Montanari. Finding one community in a sparse graph. Journal of Sta-
tistical Physics, 161(2):273–299, 2015.
[MRZ15]
Andrea Montanari, Daniel Reichman, and Ofer Zeitouni. On the limitation of
spectral methods: From the gaussian hidden clique problem to rank-one per-
turbations of gaussian tensors. In Advances in Neural Information Processing
Systems, pages 217–225, 2015.
[MW15]
Zongming Ma and Yihong Wu. Computational barriers in minimax submatrix
detection. Annals of Statistics, 43(3):1089–1116, 2015.
[RHS24]
Asaf Rotenberg, Wasim Huleihel, and Ofer Shayevitz. Planted bipartite graph
detection. IEEE Transactions on Information Theory, 2024.
[Sin18]
Amit Singer. Mathematics for cryo-electron microscopy. In Proceedings of the
International Congress of Mathematicians: Rio de Janeiro 2018, pages 3995–
4014. World Scientific, 2018.
[SN13]
Xing Sun and Andrew Nobel. On the maximal size of large-average and ANOVA-
fit submatrices in a Gaussian random matrix. Bernoulli, 19:275–294, 02 2013.
[SWP+09] Andrey A Shabalin, Victor J Weigman, Charles M Perou, Andrew B Nobel,
et al. Finding large average submatrices in high dimensional data. The Annals
of Applied Statistics, 3(3):985–1012, 2009.
37

## Page 38

[Tsy04]
Alexandre B Tsybakov. Introduction to nonparametric estimation, 2009. URL
https://doi. org/10.1007/b13794. Revised and extended from the, 9(10), 2004.
[VAC15]
Nicolas Verzelen and Ery Arias-Castro. Community detection in sparse random
networks. The Annals of Applied Probability, 25(6):3465–3510, 2015.
[WGL+16] Feng Wang, Huichao Gong, Gaochao Liu, Meijing Li, Chuangye Yan, Tian Xia,
Xueming Li, and Jianyang Zeng. DeepPicker: A deep learning approach for fully
automated particle picking in cryo-EM. Journal of structural biology, 195(3):325–
336, 2016.
38
