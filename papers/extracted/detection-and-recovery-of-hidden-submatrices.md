---
source_pdf: papers/Detection and Recovery of Hidden Submatrices.pdf
slug: detection-and-recovery-of-hidden-submatrices
pages: 36
extracted_on: 2026-07-29
---

# Detection and Recovery of Hidden Submatrices

## Page 1

arXiv:2306.06643v2  [cs.IT]  4 Jul 2023
Detection and Recovery of Hidden Submatrices
Marom Dadon
Wasim Huleihel
Tamir Bendory
July 6, 2023
Abstract
In this paper, we study the problems of detection and recovery of hidden subma-
trices with elevated means inside a large Gaussian random matrix. We consider two
diﬀerent structures for the planted submatrices. In the ﬁrst model, the planted ma-
trices are disjoint, and their row and column indices can be arbitrary. Inspired by
scientiﬁc applications, the second model restricts the row and column indices to be
consecutive. In the detection problem, under the null hypothesis, the observed matrix
is a realization of independent and identically distributed standard normal entries. Un-
der the alternative, there exists a set of hidden submatrices with elevated means inside
the same standard normal matrix. Recovery refers to the task of locating the hidden
submatrices. For both problems, and for both models, we characterize the statistical
and computational barriers by deriving information-theoretic lower bounds, designing
and analyzing algorithms matching those bounds, and proving computational lower
bounds based on the low-degree polynomials conjecture. In particular, we show that
the space of the model parameters (i.e., number of planted submatrices, their dimen-
sions, and elevated mean) can be partitioned into three regions: the impossible regime,
where all algorithms fail; the hard regime, where while detection or recovery are sta-
tistically possible, we give some evidence that polynomial-time algorithm do not exist;
and ﬁnally the easy regime, where polynomial-time algorithms exist.
1
Introduction
This paper studies the detection and recovery problems of hidden submatrices inside a large
Gaussian random matrix. In the detection problem, under the null hypothesis, the observed
matrix is a realization of an independent and identically distributed random matrix with
standard normal entries. Under the alternative, there exists a set of hidden submatrices
with elevated means inside the same standard normal matrix.
Our task is to design a
statistical test (i.e., an algorithm) to decide which hypothesis is correct. The recovery task is
M. D., W. H., and T. B. are with the Department of Electrical Engineering-Systems at Tel
Aviv university, Tel Aviv 6997801, Israel (e-mails: marom.dadon@gmail.com, wasimh@tauex.tau.ac.il,
bendory@tauex.tau.ac.il ). W.H. is supported by ISF under Grant 1734/21. T.B. is supported in part
by BSF under Grant 2020159, in part by NSF-BSF under Grant 2019752, and in part by ISF under Grant
1924/21.
1

## Page 2

the problem of locating the hidden submatrices. In this case, the devised algorithm estimates
the location of the submatrices.
We consider two statistical models for the planted submatrices. In the ﬁrst model, the
planted matrices are disjoint, and their row and column indices can be arbitrary. The de-
tection and recovery variants of this model are well-known as the submatrix detection and
submatrix recovery (or localization) problems, respectively, and received signiﬁcant attention
in the last few years, e.g., [SWP+09, KBRS11, BKR+11, BI13, ACV14, HWX15, MRZ15,
VAC15, MW15, SN13, ACCD10, BDN17, CX16, CLR17, BBH18, BBH19, Hul22], and ref-
erences therein. Speciﬁcally, for the case of a single planted submatrix, the task is to detect
the presence of a small k × k submatrix with entries sampled from a distribution P in an
n×n matrix of samples from a distribution Q. In the special case where P and Q are Gaus-
sians, the statistical and computational barriers, i.e., information-theoretic lower bounds,
algorithms, and computational lower bounds, were studied in great detail and were charac-
terized in [BI13, MRZ15, SWP+09, KBRS11, BKR+11, MW15, BBH19]. When P and Q
are Bernoulli random variables, the detection task is well-known as the planted dense sub-
graph problem, which has also been studied extensively in the literature, e.g., [BI13, ACV14,
VAC15, HWX15, BBH18]. Most notably, for both the Gaussian and Bernoulli problems, it is
well understood by now that there appears to be a statistical-computational gap between the
minimum value of k at which detection can be solved, and the minimum value of k at which
detection can be solved in polynomial time (i.e., with an eﬃcient algorithm). The statistical
and computational barriers to the recovery problem have also received signiﬁcant attention
in the literature, e.g., [CX16, Mon15, CC18, HWX16, HWX17, CLR17, BBH18], covering
several types of distributions, as well as single and (non-overlapping) multiple planted sub-
matrices.
The submatrix model above, where the planted column and row indices are arbitrary,
might be less realistic in certain scientiﬁc and engineering applications. Accordingly, we also
analyze a second model that restricts the row and column indices to be consecutive. One im-
portant motivation for this model stems from single-particle cryo-electron microscopy (cryo-
EM): a leading technology to elucidate the three-dimensional atomic structure of macro-
molecules, such as proteins [BMS15, Lyu19]. At the beginning of the algorithmic pipeline
of cryo-EM, it is required to locate multiple particle images (tomographic projections of
randomly oriented copies of the sought molecular structure) in a highly noisy, large im-
age [Sin18, BBS20].
This task is dubbed particle picking.
While many particle picking
algorithms were designed, e.g., [WGL+16, HAS18, BMR+19, ELS20], this work can be seen
as a ﬁrst attempt to unveil the statistical and computational properties of this task that
were not analyzed heretofore.
Main contributions.
To present our results, let us introduce a few notations. In our mod-
els, we have m disjoint k×k submatrices planted in an n×n matrix. We denote the observed
matrix by X. As mentioned above, we deal with the Gaussian setting, where the entries of
the planted submatrices are independent Gaussian random variables with mean λ > 0 and
unit variance, while the entries of the other entries in X are independent Gaussian random
variables with zero mean and unit variance. This falls under the general “signal+noise”
model, in the sense that X = λ · S + Z, with S being the signal of interest, Z is a Gaussian
noise matrix, and λ describes the signal-to-noise ratio (SNR) of the problem. As mentioned
2

## Page 3

above, in this paper, we consider two models for S; the ﬁrst with the arbitrary placement of
the m planted submatrices, and the second with each of the m planted submatrices having
consecutive row and column indices.
We will refer to the detection/recovery of the ﬁrst
model as submatrix detection/recovery, while for the second as consecutive submatrix de-
tection/recovery. Contrary to the consecutive submatrix detection and recovery problems,
which were not studied in the literature, the non-consecutive submatrix detection and recov-
ery problems received signiﬁcant attention; our contribution in this paper to this problem
is the analysis of the detection of multiple (possibly growing) number of planted subma-
trices, which seems to be overlooked in the literature. As mentioned above, the recovery
counterpart of multiple planted submatrices was studied in, e.g., [CLR17].
For the submatrix detection, the consecutive submatrix detection, and the consecutive
submatrix recovery problems, we study the computational and statistical boundaries and
derive information-theoretic lower bounds, algorithmic upper bounds, and computational
lower bounds. In particular, we show that the space of the model parameters (k, m, λ) can
be partitioned into diﬀerent disjoint regions: the impossible regime, where all algorithms fail;
the hard regime, where while detection or recovery are statistically possible, we give some
evidence that polynomial-time algorithms do not exist; and ﬁnally the easy regime, where
polynomial-time algorithms exist.
Table 1 summarizes the statistical and computational
thresholds for the detection and recovery problems discussed above. We emphasize that the
bounds in the second row of Table 1 (submatrix recovery), as well as the ﬁrst row (submatrix
detection) for m = 1, are known in the literature, as mentioned above.
Type
Impossible
Hard
Easy
SD
λ ≪
n
mk2 ∧
1
√
k
n
mk2 ∧
1
√
k ≪λ ≪1 ∧
n
mk2
λ ≫1 ∧
n
mk2
SR
λ ≪
1
√
k
1
√
k ≪λ ≪1 ∧
√n
k
λ ≫1 ∧
√n
k
CSD
λ ≪1
k
NO
λ ≫1
k
CSR
λ ≪
1
√
k
NO
λ ≫
1
√
k
Table 1: Statistical and computational thresholds for submatrix detection (SD), submatrix
recovery (SR), consecutive submatrix detection (CSD), and consecutive submatrix recovery
(CSR), up to poly-log factors. The bounds in the ﬁrst row for the special case of m = 1 and
the second row, are known in the literature (e.g., [BI13, MW15, CX16, CLR17]).
Interestingly, while it is well-known that the number of planted submatrices m does
not play any signiﬁcant role in the statistical and computational barriers in the submatrix
recovery problem, it can be seen that this is not the case for the submatrix detection problem.
Similarly to the submatrix recovery problem (and of course the single planted submatrix
detection problem), the submatrix detection problem undergoes a statistical-computational
gap. To provide evidence for this phenomenon, we follow a recent line of work [HS17, HB18,
BKW20, CHK+20, GJW20] and show that the class of low-degree polynomials fail to solve
the detection problem in this conjecturally hard regime. Furthermore, it can be seen that
the consecutive submatrix detection and recovery problems are either impossible or easy to
solve, namely, there is no hard regime. Here, for both the detection and recovery problems,
the number of planted submatrices m does not play an inherent role. We note that there
is a statistical gap between consecutive detection and recovery; the former is statistically
3

## Page 4

easier. This is true as long as exact recovery is the performance criterion. We also analyze
the correlated recovery (also known as weak recovery) criterion, where recovery is successful
if only a fraction of planted entries are recovered. For this weaker criterion, we show that
recovery and detection are asymptotically equivalent.
Notation.
Given a distribution P, let P⊗n denote the distribution of the n-dimensional
random vector (X1, X2, . . . , Xn), where the Xi are i.i.d. according to P. Similarly, P⊗m×n
denotes the distribution on Rm×n with i.i.d. entries distributed as P.
Given a ﬁnite or
measurable set X , let Unif[X ] denote the uniform distribution on X . The relation X ⊥⊥Y
means that the random variables X and Y are statistically independent. The Hadamard
and inner product between two n × n matrices A and B are denoted, respectively, by A ⊙B
and ⟨A, B⟩= trace(AT B). For x ∈R, we deﬁne [x]+ = max(x, 0). The nuclear norm of a
symmetric matrix A is denoted by ∥A∥⋆, and equals the summation of the absolute values of
the eigenvalues of A.
Let N (µ, σ2) denote a normal random variable with mean µ and variance σ2, when µ ∈R
and σ ∈R≥0. Let N (µ, Σ) denote a multivariate normal random vector with mean µ ∈Rd
and covariance matrix Σ, where Σ is a d × d positive semideﬁnite matrix. Let Φ denote
the cumulative distribution of a standard normal random variable with Φ(x) =
R x
−∞e−t2/2dt.
For probability measures P and Q, let dTV(P, Q) = 1
2
R
|dP −dQ|, χ2(P||Q) =
R (dP−dQ)2
dQ
,
and dKL(P||Q) = EP log dP
dQ, denote the total variation distance, the χ2-divergence, and the
Kullback-Leibler (KL) divergence, respectively. Let Bern(p) and Binomial(n, p) denote the
Bernoulli and Binomial distributions with parameters p and n, respectively. We denote by
Hypergeometric(n, k, m) the Hypergeometric distribution with parameters (n, k, m).
We use standard asymptotic notation. For two positive sequences {an} and {bn}, we
write an = O(bn) if an ≤Cbn, for some absolute constant C and for all n; an = Ω(bn),
if bn = O(an); an = Θ(bn), if an = O(bn) and an = Ω(bn), an = o(bn) or bn = ω(an), if
an/bn →0, as n →∞. Finally, for a, b ∈R, we let a ∨b ≜max{a, b} and a ∧b ≜min{a, b}.
Throughout the paper, C refers to any constant independent of the parameters of the problem
at hand and will be reused for diﬀerent constants. The notation ≪refers to polynomially
less than in n, namely, an ≪bn if lim infn→∞logn an < lim infn→∞logn bn, e.g., n ≪n2, but
n ̸≪n log2 n. For n ∈N, we let [n] = {1, 2, . . . , n}. For a subset S ⊆R, we let
1{S} denote
the indicator function of the set S.
2
Problem Formulation
In this section, we present our model and deﬁne the detection and recovery problems we
investigate, starting with the former. For simplicity of notations, we denote Q = N (0, 1)
and P = N (λ, 1), for some λ > 0, which can be interpreted as the signal-to-noise ratio
(SNR) parameter of the underlying model.
2.1
The detection problem
Let (m, k, n) be three natural numbers, satisfying m · k ≤n. We emphasize that the values
of m, k, and λ, are allowed to be functions of n—the dimension of the observation. Let
4

## Page 5

Figure 1: Illustration of the models considered in this paper: Kk,m,n of Deﬁnition 1 (left)
and Kcon
k,m,n of Deﬁnition 2 (right), for k = 4, m = 2, and n = 16.
Kk,m,n denote all possible sets that can be represented as a union of m disjoint subsets of
[n], each of size k; see Figure 1 for an illustration. Formally,
Kk,m,n ≜

Kk,m =
m
[
i=1
Si × Ti : Si, Ti ⊂Ck, ∀i ∈[m],
(Si × Ti) ∩(Sj × Tj) = ∅, ∀i ̸= j ∈[m]

,
(1)
where Ck ≜{S ⊂[n] : |S| = k}, namely, it is the set of all subsets of [n] of size k. We next
formulate two diﬀerent detection problems that we wish to investigate, starting with the
following one, a generalization of the Gaussian planted clique problem (or, bi-clustering, see,
e.g., [MW15]) to multiple hidden submatrices (or, clusters).
Deﬁnition 1 (Submatrix detection). Let (P, Q) be a pair of distributions over a measurable
space (R, B). Let SD(n, k, m, P, Q) denote the hypothesis testing problem with observation
X ∈Rn×n and hypotheses
H0 : X ∼Q⊗n×n
vs.
H1 : X ∼D(n, k, m, P, Q),
(2)
where D(n, k, m, P, Q) is the distribution of matrices X with entries Xij ∼P if i, j ∈Kk,m and
Xij ∼Q otherwise that are conditionally independent given Kk,m, which is chosen uniformly
at random over all subsets of Kk,m,n.
To wit, under H0 the elements of X are all distributed i.i.d. according to Q, while under
H1, there are m planted disjoint submatrices Kk,m in X with entries distributed according
to P, and the other entries (outside of Kk,m) are distributed according to Q.
Note that the columns and row indices of the planted submatrices in (1) can appear ev-
erywhere; in particular, they are not necessarily consecutive. In some applications, however,
we would like those submatrices to be deﬁned by a set of consecutive rows and a set of con-
secutive columns (e.g., when those submatrices model images like in cryo-EM). Accordingly,
5

## Page 6

we consider the following set:
Kcon
k,m,n ≜

Kk,m =
m
[
i=1
Si × Ti : Si, Ti ⊂Ccon
k , ∀i ∈[m],
(Si × Ti) ∩(Sj × Tj) = ∅, ∀i ̸= j ∈[m]

,
(3)
where Ccon
k
≜{S ⊂[n] : |S| = k, S is consecutive}, namely, it is the set of all subsets of [n] of
size k with consecutive elements. For example, for n = 4, we have Ccon
3
= {1, 2, 3} ∪{2, 3, 4}.
The diﬀerence between Kk,m,n and Kcon
k,m,n is depicted in Figure 1; it is evident that the
submatrices in Kk,m,n can appear everywhere, while those in Kcon
k,m,n are consecutive. Consider
the following detection problem.
Deﬁnition 2 (Consecutive submatrix detection). Let (P, Q) be a pair of distributions over
a measurable space (R, B). Let CSD(n, k, m, P, Q) denote the hypothesis testing problem with
observation X ∈Rn×n and hypotheses
H0 : X ∼Q⊗n×n
vs.
H1 : X ∼eD(n, k, m, P, Q),
(4)
where eD(n, k, m, P, Q) is the distribution of matrices X with entries Xij ∼P if i, j ∈Kk,m and
Xij ∼Q otherwise that are conditionally independent given Kk,m, which is chosen uniformly
at random over all subsets of Kcon
k,m,n.
Observing X, a detection algorithm An for the problems above is tasked with outputting
a decision in {0, 1}. We deﬁne the risk of a detection algorithm An as the sum of its Type-I
and Type-II errors probabilities, namely,
R(An) = PH0(An(X) = 1) + PH1(An(X) = 0),
(5)
where PH0 and PH1 denote the probability distributions under the null hypothesis and the
alternative hypothesis, respectively. If R(An) →0 as n →∞, then we say that An solves the
detection problem. The algorithms we consider here are either unconstrained (and thus might
be computationally expensive) or run in polynomial time (computationally eﬃcient). Typ-
ically, unconstrained algorithms are considered in order to show that information-theoretic
lower bounds are asymptotically tight. An algorithm that runs in polynomial time must
run in poly(n) time, where n is the size of the input. As mentioned in the introduction, our
goal is to derive necessary and suﬃcient conditions for when it is impossible and possible
to detect the underlying submatrices, with and without computational constraints, for both
the SD and CSD models.
2.2
The recovery problem
Next, we consider the recovery variant of the problem in Deﬁnition 2. Note that the subma-
trix recovery problem that corresponds to the problem in Deﬁnition 1, where the entries of
the submatrices are not necessarily consecutive, was investigated in [CX16]. In the recovery
problem, we assume that the data follow the distribution under H1 in Deﬁnition 2, and the
6

## Page 7

inference task is to recover the location of the planted submatrices. This is the analog of
the particle picking problem in cryo-EM that was introduced in Section 1. Consider the
following deﬁnition.
Deﬁnition 3 (Consecutive submatrix recovery). Let (P, Q) be a pair of distributions over a
measurable space (R, B). Assume that X ∈Rn×n ∼eD(n, k, m, P, Q), where eD(n, k, m, P, Q)
is the distribution of matrices X with entries Xij ∼P if i, j ∈K⋆and Xij ∼Q otherwise
that are conditionally independent given K⋆∈Kcon
k,m,n. The goal is to recover the hidden
submatrices K⋆, up to a permutation of the submatrices indices, given the matrix X. We let
CSR(n, k, m, P, Q) denote this recovery problem.
Several metrics of reconstruction accuracy are possible, and we will focus on two: exact
and correlated recovery criteria. Our estimation procedures produce a set ˆK = ˆK(X) aimed
to estimate at best the underlying true submatrices K⋆. Consider the following deﬁnitions.
Deﬁnition 4 (Exact recovery). We say that ˆK achieves exact recovery of K⋆, if, as n →∞,
supK⋆∈Kcon
k,m,n P(ˆK ̸= K⋆) →0.
Deﬁnition 5 (Correlated recovery). The overlap of K⋆and ˆK is deﬁned as the expected size
of their intersection, i.e.,
overlap(K⋆, ˆK) ≜E⟨K⋆, ˆK⟩=
n
X
i=1
P(i ∈K⋆∩ˆK).
(6)
We say that ˆK achieves correlated recovery of K⋆if there exists a ﬁxed constant ǫ > 0, such
that limn→∞supK⋆∈Kcon
k,m,n
overlap(K⋆,ˆK)
mk2
≥ǫ.
Similarly to the detection problem, also here we will care about both unconstrained and
polynomial time algorithms, and we aim to derive necessary and suﬃcient conditions for
when it is impossible and possible to recover the underlying submatrices.
3
Main Results
In this section, we present our main results for the detection and recovery problems, starting
with the former. For both problems, we derive the statistical and computational bounds for
the two models we presented in the previous section.
3.1
The detection problem
Upper bounds.
We start by presenting our upper bounds. To that end, we propose three
algorithms and analyze their performance. Deﬁne the statistics,
Tsum(X) ≜
X
i,j∈[n]
Xij,
(7)
TSD
scan(X) ≜
max
K∈Kk,1,n
X
i,j∈K
Xij,
(8)
7

## Page 8

TCSD
scan(X) ≜
max
K∈Kcon
k,1,n
X
i,j∈K
Xij.
(9)
The statistics in (7) amounts to adding up all the elements of X, while (8) and (9) enumerate
all k × k submatrices of X in Kk,1,n and Kcon
k,1,n, and take the submatrix with the maximal
sum of entries, respectively. Fix δ > 0. Then, our tests are deﬁned as,
Asum(X) ≜
1 {Tsum(X) ≥τsum} ,
(10)
ASD
scan(X) ≜
1

TSD
scan(X) ≥τ SD
scan
	
,
(11)
ACSD
scan(X) ≜
1

TCSD
scan(X) ≥τ CSD
scan
	
,
(12)
where the thresholds are given by τsum ≜
mk2λ
2
, τ SD
scan ≜
q
(4 + δ)k2 log
 n
k

, and τ CSD
scan ≜
p
(4 + δ)k2 log n, and correspond roughly to the average between the expected values of
each of the statistics in (7)–(9) under the null and alternative hypotheses.
It should be
emphasized that the tests in (10)–(11) were proposed in, e.g., [KBRS11, BI13, MW15], for
the single planted submatrix detection problem.
A few important remarks are in order. First, note that in the scan test, we search for a
single planted matrix rather than m such matrices. Second, the sum test exhibits polynomial
computational complexity, of O(n2) operations, and hence eﬃcient. The scan test in (11),
however, exhibits an exponential computational complexity, and thus is ineﬃcient. Indeed,
the search space in (11) is of cardinality |Kk,1,n| =
 n
k
2.
On the other hand, the scan
test ACSD
scan for the consecutive setting is eﬃcient because |Kcon
k,1,n| ≤n2. The following result
provides suﬃcient conditions under which the risk of each of the above tests is asymptotically
small.
Theorem 1 (Detection upper bounds). Consider the detection problems in Deﬁnitions 1
and 2. Then, we have the following bounds:
1. (Eﬃcient SD) There exists an eﬃcient algorithm Asum in (10), such that if
λ = ω
 n
mk2

,
(13)
then R (Asum) →0, as n →∞, for the problems in Deﬁnitions 1 and 2.
2. (Exhaustive SD) There exists an algorithm ASD
scan in (11), such that if
λ = ω
 r
log n
k
k
!
,
(14)
then R
 ASD
scan

→0, as n →∞, for the problem in Deﬁnition 1.
3. (Eﬃcient CSD) There exists an eﬃcient algorithm ACSD
scan in (12), such that if
λ = ω
 plog n
k
k
!
,
(15)
then R(ACSD
scan) →0, as n →∞, for the problem in Deﬁnition 2.
8

## Page 9

As can be seen from Theorem 1, only the sum test performance barrier exhibits depen-
dency on m. The scan test is, for both SD and CSD, inherently independent of m. This
makes sense because when summing all the elements of X, as m gets larger the mean (the
“signal”) under the alternative hypothesis gets larger as well. On the other hand, since the
scan test searches for a single planted submatrix, the number of planted submatrices does
not play a role. One could argue that it might be beneﬁcial to search for the m planted
submatrices in the scan test, however, as we show below, this is not needed, and the bounds
above are asymptotically tight.
Lower bounds.
To present our lower bounds, we ﬁrst recall that the optimal testing error
probability is determined by the total variation distance between the distributions under the
null and the alternative hypotheses as follows (see, e.g., [Tsy08, Lemma 2.1]),
min
An:Rn×n→{0,1} PH0(An(X) = 1) + PH1(An(X) = 0) = 1 −dTV(PH0, PH1).
(16)
The following result shows that under certain conditions the total variation between the null
and alternative distributions is asymptotically small, and thus, there exists no test which
can solve the above detection problems reliably.
Theorem 2 (Information-theoretic lower bounds). We have the following results.
1. Consider the detection problem in Deﬁnition 1. If,
λ = o
 n
mk2 ∧1
√
k

,
(17)
then dTV(PH0, PH1) = o(1).
2. Consider the detection problem in Deﬁnition 2. If λ = o (k−1), then dTV(PH0, PH1) =
o(1).
Theorem 2 above shows that our upper bounds in Theorem 1 are tight up to poly-
log factors. Indeed, item 1 in Theorem 2 complements Items 1-2 in Theorem 1, for the
SD problem, while item 2 in Theorem 2 complements Item 3 in Theorem 1, for the CSD
problem. In the sequel, we illustrate our results using phase diagrams that show the tradeoﬀ
between k and λ as a function of n. One evident and important observation here is that
the statistical limit for the CSD problem is attained using an eﬃcient test. Thus, there is
no statistical computational gap in the detection problem in Deﬁnition 2, and accordingly,
it is either statistically impossible to solve the detection problem or it can be solved in
polynomial time. This is not the case for the SD problem. Note that both the eﬃcient sum
and the exhaustive scan tests are needed to attain the information-theoretic lower bound
(up to poly-log factors). As discussed above, however, here the scan test is not eﬃcient. We
next give evidence that, based on the low-degree polynomial conjecture, eﬃcient algorithms
that run in polynomial-time do not exist in the regime where the scan test succeeds while
the sum test fails.
9

## Page 10

Computational lower bounds.
Note that the problem in Deﬁnition 1 exhibits a gap in
terms of what can be achieved by the proposed polynomial-time algorithm and the compu-
tationally expensive scan test algorithm. In particular, it can be seen that in the regime
where
1
√
k ≪λ ≪
n
mk2, while the problem can be solved by an exhaustive search using the
scan test, we do not have a polynomial-time algorithm. Next, we give evidence that, in
fact, an eﬃcient algorithm does not exist in this region. To that end, we start with a brief
introduction to the method of low-degree polynomials.
The premise of this method is to take low-degree multivariate polynomials in the entries
of the observations as a proxy for eﬃciently-computable functions. The ideas below were ﬁrst
developed in a sequence of works in the sum-of-squares optimization literature [BHK+16,
HB18, HS17, HKP+17].
In the following, we follow the notations and deﬁnitions of [HB18, BPW18]. Any distri-
bution PH0 on Ωn = Rn×n induces an inner product of measurable functions f, g : Ωn →R
given by ⟨f, g⟩H0 = EH0[f(X)g(X)], and norm ∥f∥H0 = ⟨f, f⟩1/2
H0 . We Let L2(PH0) denote the
Hilbert space consisting of functions f for which ∥f∥H0 < ∞, endowed with the above inner
product and norm. In the computationally-unbounded case, the Neyman-Pearson lemma
shows that the likelihood ratio test achieves the optimal tradeoﬀbetween Type-I and Type-II
error probabilities. Furthermore, it is well-known that the same test optimally distinguishes
PH0 from PH1 in the L2 sense. Speciﬁcally, denoting by Ln ≜PH1/PH0 the likelihood ratio,
the second-moment method for contiguity (see, e.g., [BPW18]) shows that if ∥Ln∥2
H0 remains
bounded as n →∞, then PH1 is contiguous to PH0. This implies that PH1 and PH0 are sta-
tistically indistinguishable, i.e., no test can have both Type-I and Type-II error probabilities
tending to zero.
We now describe the low-degree method. The idea is to ﬁnd the low-degree polynomial
that best distinguishes PH0 from PH1 in the L2 sense. To that end, we let Vn,≤D ⊂L2(PH0)
denote the linear subspace of polynomials Ωn →R of degree at most D ∈N. We further
deﬁne P≤D : L2(PH0) →Vn,≤D the orthogonal projection operator. Then, the D-low-degree
likelihood ratio L≤D
n
is the projection of a function Ln to the span of coordinate-degree-
D functions, where the projection is orthogonal with respect to the inner product ⟨·, ·⟩H0.
As discussed above, the likelihood ratio optimally distinguishes PH0 from PH1 in the L2
sense. The next lemma shows that over the set of low-degree polynomials, the D-low-degree
likelihood ratio have exhibit the same property.
Lemma 1 (Optimally of L≤D
n
[HS17, HKP+17, BPW18]). Consider the following optimiza-
tion problem:
max EH1f(X)
s.t.
EH0f 2(X) = 1, f ∈Vn,≤D.
(18)
Then, the unique solution f ⋆for (18) is the D-low degree likelihood ratio f ⋆= L≤D
n /
L≤D
n

H0,
and the value of the optimization problem is
L≤D
n

H0.
As was mentioned above, in the computationally-unbounded regime, an important prop-
erty of the likelihood ratio is that if ∥Ln∥H0 is bounded, then PH0 and PH1 are statistically
indistinguishable. The following conjecture states that a computational analog of this prop-
erty holds, with L≤D
n
playing the role of the likelihood ratio. In fact, it also postulates that
polynomials of degree ≈log n are a proxy for polynomial-time algorithms. The conjecture
below is based on [HB18, HS17, HKP+17], and [HB18, Conj. 2.2.4]. We give an informal
10

## Page 11

statement of this conjecture, which appears in [BPW18, Conj. 1.16]. For a precise statement,
we refer the reader to [HB18, Conj. 2.2.4] and [BPW18, Sec. 4].
Conjecture 1 (Low-degree conjecture, informal). Given a sequence of probability measures
PH0 and PH1, if there exists ǫ > 0 and D = D(n) ≥(log n)1+ǫ, such that
L≤D
n

H0 remains
bounded as n →∞, then there is no polynomial-time algorithm that distinguishes PH0 and
PH1.
In the sequel, we will rely on Conjecture 1 to give evidence for the statistical-
computational gap observed for the problem in Deﬁnition 1 in the regime where
1
√
k ≪
λ ≪
n
mk2. At this point we would like to mention [HB18, Hypothesis 2.1.5], which states a
more general form of Conjecture 1 in the sense that it postulates that degree-D polynomi-
als are a proxy for nO(D)-time algorithms. Note that if
L≤D
n

H0 = O(1), then we expect
detection in time T(n) = eD(n) to be impossible.
Theorem 3 (Computational lower bound). Consider the detection problem in Deﬁnition 1.
Then, if λ is such that
1
√
k ≪λ ≪
n
mk2, then
L≤D
n

H0 ≤O(1), for any D = Ω(log n). On
the other hand, if λ is such that λ ≫
n
mk2, then
L≤D
n

H0 ≥ω(1).
Together with Conjecture 1, Theorem 3 implies that if we take degree-log n polynomials
as a proxy for all eﬃcient algorithms, our calculations predict that an nO(log n) algorithm
does not exist when
1
√
k ≪λ ≪
n
mk2. This is summarized in the following corollary.
Corollary 4. Consider the detection problem in Deﬁnition 1, and assume that Conjecture 1
holds. An nO(log n) algorithm that achieves strong detection does not exist if λ is such that
1
√
k ≪λ ≪
n
mk2.
These
predictions
agree
precisely
with
the
previously
established
statistical-
computational tradeoﬀs in the previous subsections. A more explicit formula for the com-
putational barrier which exhibits dependency on D and λ can be deduced from the proof of
Theorem 3; to keep the exposition simple we opted to present the reﬁned result above.
We note that numerical and theoretical evidence for the existence of computational-
statistical gaps were observed in other statistical models that are also inspired by cryo-
EM, including heterogeneous multi-reference alignment [BBLS18, Wei18] and sparse multi-
reference alignment [BMS22].
Phase diagrams.
Using Theorems 1–3 we are now in a position to draw the obtained
phase diagrams for our detection problems. Speciﬁcally, treating k and λ as polynomials
in n, i.e., k = Θ(nβ) and λ = Θ(n−α), for some α ∈(0, 1) and β ∈(0, 1), we obtain the
phase diagrams in Figure 2a, for a ﬁxed number of submatrices m = O(1). Speciﬁcally,
1. Computationally easy regime (blue region): there is a polynomial-time algorithm for
the detection task when α < 2β −1.
2. Computationally hard regime (red region): there is an ineﬃcient algorithm for detection
when α < β/2 and α > 2β−1, but the problem is computationally hard (no polynomial-
time algorithm exists) in the sense that the class of low-degree polynomials fails in this
region.
11

## Page 12

β
α
1
1
2
1
1
3
2
3
0
“Statistically
Impossible”
“Easy”
“Hard”
(a) m = O(1)
β
α
1
3
8
5
4
1
4
1
2
0
“Statistically
Impossible”
“Easy”
“Hard”
(b) m = Θ(n1/4)
Figure 2: Phase diagrams for submatrix detection as a function of k = Θ(nβ), and λ =
Θ(n−α), for m = O(1) and m = Θ(n1/4).
3. Statistically impossible regime: detection is statistically impossible when α > β
2 ∨(2β −
1).
When the number of submatrices grows with n = ω(1), we get diﬀerent phase diagrams
depending on its value. For example, if m = Θ(n1/4), we get Figure 2b. Speciﬁcally,
1. Computationally easy regime (blue region): there is a polynomial-time algorithm for
the detection task when α < 2β −3
4.
2. Computationally hard regime (red region): there is an ineﬃcient algorithm for de-
tection when α < β/2 and α > 2β −3
4, but the problem is computationally hard (no
polynomial-time algorithm exists) in the sense that the class of low-degree polynomials
fails in this region.
3. Statistically impossible regime: detection is statistically impossible when α > β
2 ∨(2β −
3/4).
Finally, for the consecutive problem, we get the phase diagram in Figure 3, independently
of the value of m. Here, there are only two regions where the problem is either statistically
impossible or easy to solve.
3.2
The recovery problem
Upper bounds.
We start by presenting our upper bounds for both exact and correlated
types of recovery for the consecutive problem in Deﬁnition 3. To that end, we propose the
12

## Page 13

β
α
1
1
0
“Statistically
Impossible”
“Easy”
Figure 3: Phase diagram for consecutive submatrix detection, as a function of k = Θ(nβ),
and λ = Θ(n−α), for any m.
following recovery algorithm. It can be shown that the maximum-likelihood (ML) estimator,
minimizing the error probability, is given by (see Subsection 4.4 for a complete derivation),
ˆKML(X) = arg max
K∈Kcon
k,m,n
X
(i,j)∈K
Xij.
(19)
The computational complexity of the exhaustive search in (19) is of order n2m. Thus, for
m = O(1), the ML estimator runs in polynomial time, and thus, is eﬃcient. However, if
m = ω(1) then the exhaustive search is not eﬃcient anymore. Nonetheless, the following
straightforward modiﬁcation of (19) provably achieves the same asymptotic performance of
the ML estimator above, and at the same time computationally eﬃcient.
Before we present this algorithm, we make a simplifying technical assumption on the
possible set of planted submatrices, and then explain how this assumption can be removed.
We assume that each pair of submatrices in the underlying planted submatrices K⋆are at
least k columns and rows far way. In other words, there are at least k columns and k rows
separating any pair of submatrices in K⋆. Similar assumptions are frequently taken when
analyzing statistical models inspired by cryo-EM, see, for example [BBL+18]. We will refer
to the above as the separation assumption.
Our recovery algorithm works as follows: in the ℓ∈[m] step, we ﬁnd the ML estimate
of a single submatrix using,
ˆKℓ(X(ℓ)) = arg max
K∈Kcon
k,1,n
X
(i,j)∈K
X(ℓ)
ij ,
(20)
where X(ℓ) is deﬁned recursively as follows: X(1) ≜X, and for ℓ≥2,
X(ℓ) = X(ℓ−1) ⊙E(ˆKℓ−1),
(21)
where E(ˆKℓ−1) is an n × n matrix such that [E(ˆKℓ−1)]ij = −∞, for (i, j) ∈ˆKℓ−1, and
[E(ˆKℓ−1)]ij = 1, otherwise.
To wit, in each step of the algorithm we “peel” the set of
13

## Page 14

estimated indices (or, estimated submatrices) in previous steps from the search space. This
is done by setting the corresponding entries of X to −∞so that the sum in (20) will not be
maximized by previously chosen sets of indices. We denote by ˆKpeel(X) = {ˆKℓ}m
ℓ=1 the output
of the above algorithm.
Remark 1. Without the assumption above, the fact that the peeling algorithm succeeds is
not trivial. If, for example, the chosen planted matrices are such that they include a pair
of adjacent matrices, then it could be the case that at some step of the peeling algorithm,
the estimated set of indices corresponds to a certain submatrix of the union of those adja-
cent matrices. However, one can easily modify the peeling algorithm, drop the assumption
above, and obtain the same statistical guarantees stated below. Indeed, consider the following
modiﬁcation to the peeling routine in Algorithm 1.
Algorithm 1 Modified Peeling
1. Initialize ﬂag ←0, ℓ←1, K ←∅, A = 0n×n.
2. while ﬂag = 0
(a) ˆKℓ(X) ←arg maxK∈Kcon
k,1,n\K
P
(i,j)∈K Xij.
(b) Aij ←1, for (i, j) ∈ˆKℓ(X), and Aij ←0, otherwise.
(c) K ←K ∪ˆKℓ(X).
(d) if ⟨J, A⟩= mk2
ﬂag ←1.
(e) else
ℓ←ℓ+ 1.
3. Output A.
The key idea is as follows. In the ﬁrst step, we ﬁnd the k × k submatrix in X with the
maximum sum of entries. We denote this submatrix by ˆK1. This is exactly the same ﬁrst
step of the peeling algorithm. In the second step, we again search for the k×k submatrix in X
with the maximum sum of entries, but of course, remove ˆK1 from the search space. More
generally, in the ℓ-th step, we again search for the k × k submatrix in X with maximum sum
of entries, but remove K = ∪ℓ−1
i=1 ˆKi from the search space. We terminate this process once
∪ℓ
i=1ˆKi ∈Kcon
k,m,n, i.e., the union of the estimated sets of matrices can cast as a proper set of
planted submatrices. This can easily be checked by forming the matrix A in Step 2(b), and
checking the conditions in Step 2(d). If the actually planted submatrices are not adjacent,
then this will be the case (under the conditions in the theorem below) after ℓ= m steps, with
high probability. Otherwise, if at least two planted submatrices are adjacent, then while ℓ
might be larger than m it is bounded by n2, and it is guaranteed that such a union exists.
Once we ﬁnd such a union, it is easy to revert the set of m consecutive k × k submatrices
from A.
We have the following result.
Theorem 5 (Recovery upper bounds). Consider the recovery problem in Deﬁnition 3, and
let C be a universal constant. Then, we have the following set of bounds:
14

## Page 15

1. (ML Exact Recovery) Consider the ML estimator in (19). If
lim inf
n→∞
λ
p
Ck−1 log n
> 1,
(22)
then exact recovery is possible.
2. (Peeling Exact Recovery) Consider the peeling estimator in (20), and assume that the
separation assumption holds. Then, if
lim inf
n→∞
λ
p
Ck−1 log n
> 1,
(23)
then exact recovery is possible.
3. (Peeling Correlated Recovery) Consider the peeling estimator in (20), and assume that
the separation assumption holds. If
lim inf
n→∞
λ
p
Ck−2 log n
> 1,
(24)
then correlated recovery is possible.
Lower bounds.
The following result shows that under certain conditions, exact and cor-
related recoveries are impossible.
Theorem 6 (Information-theoretic recovery lower bounds). Consider the recovery problem
in Deﬁnition 3. Then:
1. If λ < C
q
log m
k , exact recovery is impossible, i.e.,
inf
ˆK
sup
K⋆∈Kcon
k,m,n
P[ˆK(X) ̸= K⋆] > 1
2,
where the inﬁmum ranges over all measurable functions of the matrix X.
2. If λ = o(k−1), correlated recovery is impossible, i.e., supK⋆∈Kcon
k,m,n overlap(K⋆, ˆK) =
o(mk2).
Thus, similarly to the detection problem, the consecutive recovery problem is either
statistically impossible or easy to solve. The corresponding phase diagram for exact and
correlated types of recoveries is given in Figure 4.
Roughly speaking, exact recovery is
possible if λ = ω(k−1/2) and impossible if λ = o(k−1/2). Correlated recovery is possible if
λ = ω(k−1) and impossible if λ = o(k−1).
A few remarks are in order.
First, note that there is a gap between detection and
exact recovery; the barrier for λ for the former is at k−1, while for the latter at k−1/2. In the
context of cryo-EM, this indicates a gap between the ability to detect the existence of particle
images in the data set, and the ability to perform successful particle picking (exact recovery).
15

## Page 16

β
α
1
1
0
1
2
1
2
“Statistically
Impossible”
“Peeling”
β
α
1
1
0
1
2
1
2
“Statistically
Impossible”
“Peeling”
Figure 4: Phase diagram for consecutive submatrix exact recovery (left) and correlated
recovery (right), as a function of k = Θ(nβ), and λ = Θ(n−α), for any m.
Recently, new computational methods were devised to elucidate molecular structures without
particle picking, thus bypassing the limit of exact recovery, allowing constructing structures
in very low SNR environments, e.g., [BBL+18, KB22, KSB23]. This in turn opens the door
to recovering small molecular structures that induce low SNR [Hen95]. Second, there is no
gap between detection and correlated recovery, and these diﬀerent tasks are asymptotically
statistically the same. The same gap exists between correlated and exact recoveries, implying
that exact recovery is strictly harder than correlated recovery.
4
Proofs
4.1
Proof of Theorem 1
4.1.1
Sum test
Recall the sum test in (10), and let τ ≜
mk2λ
2
.
Let us analyze the corresponding error
probability. On the one hand, under H0, it is clear that Tsum(X) ∼N (0, n2). Thus,
PH0 (Asum(X) = 1) = PH0 (Tsum(X) ≥τ)
= P(N (0, n2) ≥τ)
(25)
≤1
2 exp

−τ 2
2n2

.
On the other hand, under H1, Tsum(X) ∼N (mk2λ, n2). Thus,
PH1 (Asum(X) = 0) = PH1 (Tsum(X) ≤τ)
= P(N (mk2λ, n2) ≤τ)
(26)
≤1
2 exp

−(τ −mk2λ)2
2n2

.
16

## Page 17

Substituting τ = mk2λ
2
, we obtain that
R (Asum) ≤exp

−m2k4λ2
8n2

.
(27)
Thus, if mk2λ
n
→∞, then R (Asum) →0, as n →∞. Note that the analysis above holds true
for both detection problems in Deﬁnitions 1 and 2.
4.1.2
Scan test
Recall the scan test ASD
scan(X) in (11), and its consecutive version ACSD
scan(X).
Let us start
by analyzing the error probability associated with ASD
scan. For simplicity of notation, we let
τ ≜
q
(4 + δ)k2 log
 n
k

. On the one hand, under H0, we have
PH0
 ASD
scan(X) = 1

= PH0
 TSD
scan(X) ≥τ

≤
n
k
2
P(N (0, k2) ≥τ)
(28)
≤1
2 exp

2 log
n
k

−τ 2
2k2

.
On the other hand, under H1, we have
PH1
 ASD
scan(X) = 0

= PH1
 TSD
scan(X) ≤τ

(29)
≤P(N (k2λ, k2) ≤τ)
(30)
≤1
2 exp

−(k2λ −τ)2
+
2k2

.
(31)
Thus,
R
 ASD
scan

≤1
2 exp

2 log
n
k

−τ 2
2k2

+ exp

−(k2λ −τ)2
+
2k2

.
(32)
Substituting τ =
q
(4 + δ)k2 log
 n
k

, we get
R
 ASD
scan

≤1
2 exp

−δ
2 · log
n
k

+ exp
 
−k2
 λ −τ
k2
2
+
2
!
,
(33)
and thus R
 ASD
scan

→0, as n →∞, provided that lim infn→∞
λ
√
4k−1 log n
k > 1, as claimed.
Next, we analyze ACSD
scan. Let τc ≜
p
(4 + δ)k2 log n. As above, we have
PH0
 ACSD
scan(X) = 1

= PH0
 TCSD
scan(X) ≥τc

(34)
≤n2P(N (0, k2) ≥τc)
(35)
≤1
2 exp

2 log n −τ 2
c
2k2

.
(36)
17

## Page 18

On the other hand, under H1, the result remained intact:
PH1
 ACSD
scan(X) = 0

= PH1
 TCSD
scan(X) ≤τc

(37)
≤P(N (k2λ, k2) ≤τc)
(38)
≤1
2 exp

−(k2λ −τc)2
+
2k2

.
(39)
Thus,
R
 ACSD
scan

≤1
2 exp

2 log n −τ 2
c
2k2

+ exp

−(k2λ −τc)2
+
2k2

.
(40)
Substituting τc =
p
(4 + δ)k2 log n, for δ > 0, we get
R
 ACSD
scan

≤1
2 exp

−δ
2 · log n

+ exp
 
−k2
 λ −τc
k2
2
+
2
!
,
(41)
and thus R
 ACSD
scan

→0, as n →∞, provided that lim infn→∞
λ
4√
k−2 log n
k > 1, as claimed.
4.2
Proof of Theorem 2
4.2.1
Submatrix detection
Recall that the optimal test A∗
n that minimizes the risk is the likelihood ratio test deﬁned
as follows,
A∗
n (X) ≜
1 {Ln (X) ≥1} ,
(42)
where Ln (X) ≜
PH1(X)
PH0(X). The optimal risk, denoted by R∗= R(A∗
n), can be lower bounded
using the Cauchy–Schwartz inequality as follows,
R∗= 1 −1
2EH0 |Ln (X) −1|
(43)
≥1 −1
2
q
EH0

(Ln (X) −1)2
= 1 −1
2
q
EH0

(Ln (X))2
−1.
Thus, in order to lower bound the risk, we need to upper bound EH0

(Ln (X))2
. Below, we
provide a lower bound that holds for any pair of distributions P and Q.
Corollary 7. The following holds:
EH0

(Ln (X))2
= EK⊥⊥K′
h
(1 + χ2(P||Q))|K∩K′|i
≤EK⊥⊥K′
h
eχ2(P||Q)·|K∩K′|i
,
(44)
where K and K′ are two independent copies drawn uniformly at random from Kk,m,n (or,
¯Kk,m,n), and
χ2(P||Q) ≜EX∼Q
P(X)
Q(X)
2
−1.
(45)
18

## Page 19

Proof of Corollary 7. First, note that the likelihood can be written as follows:
Ln (X) = PH1(X)
PH0(X) = EK∼Unif(Kk,m,n)

Y
(i,j)∈K
P(Xij)
Q(Xij)

.
(46)
Now, note that the square of the right-hand side of (46) can be rewritten as:

EK∼Unif(Kk,m,n)

Y
(i,j)∈K
P(Xij)
Q(Xij)




2
= EK⊥⊥K′∼Unif(Kk,m,n)

Y
(i,j)∈K
P(Xij)
Q(Xij)
Y
(i,j)∈K′
P(Xij)
Q(Xij)

.
(47)
Therefore,
EH0

(Ln (X))2
= EH0

EK∼Unif(Kk,m,n)

Y
(i,j)∈K
P(Xij)
Q(Xij)




2
(48)
= EH0

EK⊥⊥K′∼Unif(Kk,m,n)

Y
(i,j)∈K
P(Xij)
Q(Xij)
Y
(i,j)∈K′
P(Xij)
Q(Xij)




(49)
= EK⊥⊥K′∼Unif(Kk,m,n)

EH0

Y
(i,j)∈K
P(Xij)
Q(Xij)
Y
(i,j)∈K′
P(Xij)
Q(Xij)




(50)
= EK⊥⊥K′∼Unif(Kk,m,n)

EH0


Y
(i,j)∈K∪K′\K∩K′
P(Xij)
Q(Xij)
Y
(i,j)∈K∩K′
P(Xij)
Q(Xij)
2




(51)
= EK⊥⊥K′∼Unif(Kk,m,n)


Y
(i,j)∈K∪K′\K∩K′
EH0
P(Xij)
Q(Xij)

Y
(i,j)∈K∩K′
EH0
P(Xij)
Q(Xij)
2


(52)
(a)
= EK⊥⊥K′∼Unif(Kk,m,n)


 
EH0
P(Xij)
Q(Xij)
2!|K∩K′|

(53)
= EK⊥⊥K′∼Unif(Kk,m,n)
h 1 + χ2(P||Q)
|K∩K′|i
(54)
(b)
≤EK⊥⊥K′
h
eχ2(P||Q)·|K∩K′|i
,
(55)
where (a) is because EQ
P(Xij)
Q(Xij) = 1, and (b) is because 1 + x ≤exp(x), for any x ∈R.
Based on Corollary 7, it suﬃces to upper bound EK⊥⊥K′
h
eχ2(P||Q)·|K∩K′|i
. Recall that K
and K′ are decomposed as K = Sm
ℓ=1 Sℓ× Tℓand K′ = Sm
ℓ=1 S′
ℓ× T′
ℓ. Thus, we note that the
intersection of K and K′ can be rewritten as
|K ∩K′| =
m
X
ℓ1=1
m
X
ℓ2=1
|(Sℓ1 ∩S′
ℓ2) × (Tℓ1 ∩T′
ℓ2)|
(56)
19

## Page 20

=
m
X
ℓ1=1
m
X
ℓ2=1
|(Sℓ1 ∩S′
ℓ2)| · |(Tℓ1 ∩T′
ℓ2)|.
(57)
For each ℓ1, ℓ2 ∈[m], deﬁne Zℓ1,ℓ2 ≜|(Sℓ1 ∩S′
ℓ2)| and Rℓ1,ℓ2 ≜|(Tℓ1 ∩T′
ℓ2)|.
Note that
the sequence of random variables {Zℓ1,ℓ2}ℓ1,ℓ2 are statistically independent of the sequence
{Rℓ1,ℓ2}ℓ1,ℓ2.
Next, it is easy to show that Zℓ1,ℓ2 ∼Hypergeometric(n, k, k) and Rℓ1,ℓ2 ∼
Hypergeometric(n, k, k), for each ℓ1, ℓ2 ∈[m], for any ℓ1, ℓ2 ∈[m]. Indeed, if we have an urn
of n balls among which k balls are red, the random variable Zℓ1,ℓ2 (and Rℓ1,ℓ2) is exactly the
number of red balls if we draw k balls from the urn uniformly at random without replacement,
which is the deﬁnition of a Hypergeometric random variable. While the random variables
{Zℓ1,ℓ2}ℓ1,ℓ2 (and similarly {Rℓ1,ℓ2}ℓ1,ℓ2) are not independent, they are negatively associated.
Thus,
EK⊥⊥K′
h
eχ2(P||Q)·|K∩K′|i
≤
m
Y
ℓ1=1
m
Y
ℓ2=1
E
h
eχ2(P||Q)·Zℓ1,ℓ2Rℓ1,ℓ2
i
=
h
E

eχ2(P||Q)·Z1,1R1,1im2
.
(58)
Next,
it is well-known that Z1,1
=
Hypergeometric(n, k, k) (and similarly R1,1
=
Hypergeometric(n, k, k))
is
stochastically
dominated
by
B
∼
Binomial(k, k/n)
=
Pk
i=1 Bern(k/n). Thus,
E

eχ2(P||Q)·Z1,1R1,1
≤E

eχ2(P||Q)·BB′
,
(59)
where B′ be an independent copy of B. Thus,
EK⊥⊥K′
h
eχ2(P||Q)·|K∩K′|i
≤
h
E

eχ2(P||Q)·BB′im2
.
(60)
We show that, if χ2(P||Q) satisﬁes the condition of Theorem 2, the term on the right-hand
side of (60) is at most 1 + δ, for any δ > 0. We have
h
E

eχ2(P||Q)·BB′im2
=
"
E

1 + k
n

eχ2(P||Q)B −1
k#m2
.
(61)
Next, note that B ≤k and we also assume the following, for reasons that will become clear,
χ2(P||Q) ≤1
k.
(62)
Therefore, using the inequality ex −1 ≤x + x2, for x < 1, the following holds
h
E

eχ2(P||Q)·BB′im2
≤
"
E

1 + k
n
 χ2(P||Q)B + χ4(P||Q)B2k#m2
(63)
≤
"
E

1 + 2k
nχ2(P||Q)B
k#m2
(64)
20

## Page 21

≤
h
E

e2 k2
n χ2(P||Q)Bim2
(65)
=

1 + k
n

e2 k2
n χ2(P||Q) −1
km2
.
(66)
This is at most 1 + δ if
k
n

e2 k2
n χ2(P||Q) −1

≤(1 + δ)
1
km2 −1.
(67)
Since (1 + δ)
1
km2 −1 ≥log(1 + δ)/(km2), this is implied by
χ2(P||Q) ≤n
2k2 log

1 + n log(1 + δ)
m2k2

.
(68)
Putting altogether, we obtained that EK⊥⊥K′
h
eχ2(P||Q)·|K∩K′|i
≤1 + δ, if
χ2(P||Q) ≤min
1
k, n
2k2 log

1 + n log(1 + δ)
m2k2

(69)
= min
1
k, n2 log(1 + δ)
2m2k4

.
(70)
Finally, note that in the Gaussian case, χ2(N (λ, 1)||N (0, 1)) = 1
2 [exp (λ2) −1]. Thus, for
λ = o(1), we have χ2(N (λ, 1)||N (0, 1)) →λ2
2 , which concludes the proof.
4.2.2
Consecutive submatrix detection
For the consecutive case, we notice that by using the steps as in the previous subsection, we
have
EH0

(Ln (X))2
≤EK⊥⊥K′
h
eχ2(P||Q)·|K∩K′|i
,
(71)
where K and K′ are two independent copies drawn uniformly at random from Kcon
k,m,n. The
key distinction from the previous case lies in the distribution of |K ∩K′|. Recall that K and
K′ are decomposed as K = Sm
ℓ=1 Sℓ× Tℓand K′ = Sm
ℓ=1 S′
ℓ× T′
ℓ. Thus, we note that the
intersection of K and K′ can be rewritten as
|K ∩K′| =
m
X
ℓ1=1
m
X
ℓ2=1
|(Sℓ1 ∩S′
ℓ2) × (Tℓ1 ∩T′
ℓ2)|
(72)
=
m
X
ℓ1=1
m
X
ℓ2=1
|(Sℓ1 ∩S′
ℓ2)| · |(Tℓ1 ∩T′
ℓ2)|
(73)
≜
m
X
ℓ1=1
m
X
ℓ2=1
Zℓ1,ℓ2.
(74)
21

## Page 22

Note that for a given pair (ℓ1, ℓ2), we have
P(|(Sℓ1 ∩S′
ℓ2)| = z) =





n−2k+1
n
,
for z = 0
2
n,
for z = 1, 2, ..., k −1
1
n,
for z = k,
(75)
and the exact same distribution for |(Tℓ1 ∩T′
ℓ2)|. Thus, we may write Zℓ1,ℓ2
(d)
= H · H′, where
H and H′ are statistically independent and follow the distribution given in (75). Thus, using
the fact that the random variables {Zℓ1,ℓ2}ℓ1,ℓ2 are negatively associated, we get,
EK⊥⊥K′
h
eχ2(P||Q)·|K∩K′|i
≤
m
Y
ℓ1=1
m
Y
ℓ2=1
E
h
eχ2(P||Q)·Zℓ1,ℓ2
i
=
h
E

eχ2(P||Q)·H·H′im2
.
(76)
Now,
E

eχ2(P||Q)·H·H′
= E
 
n −2k + 1
n
+ 2
n
k−1
X
i=1
eχ2(P||Q)·iH′ + eχ2(P||Q)·kH′
n
!
(77)
≤E
n −2k
n
+ 2k
n eχ2(P||Q)·kH′
(78)
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
i=1
eχ2(P||Q)·ik + eχ2(P||Q)·k2
n
!
(79)
≤n −2k
n
+ 2k
n
n −2k
n
+ 2k
n eχ2(P||Q)·k2
(80)
= 1 + 4k2
n2

eχ2(P||Q)·k2 −1

.
(81)
Therefore,
EK⊥⊥K′
h
eχ2(P||Q)·|K∩K′|i
≤

1 + 4k2
n2

eχ2(P||Q)·k2 −1
m2
.
(82)
This is at most 1 + δ if,
4k2
n2

eχ2(P||Q)k2 −1

≤(1 + δ)
1
m2 −1.
(83)
Since (1 + δ)
1
m2 −1 ≥log(1 + δ)/(m2), this is implied by
χ2(P||Q) ≤1
k2 log

1 + n2 log(1 + δ)
4k2m2

.
(84)
Finally, note that since km ≤n, the logarithmic factor in (84) can be lower bounded by
log(1 + log(1 + δ)/4), which concludes the proof.
22

## Page 23

4.3
Proof of Theorem 3
In order to prove Theorem 3, we use the following result [BPW18, Theorem 2.6].
Lemma 2. Let S be an n dimensional random vector drawn from some distribution Dn, and
let Z be an i.i.d. n dimensional random vector with standard normal entries. Consider the
detection problem:
H0 : Y = Z
vs.
H1 : Y = S + Z.
(85)
Then,
L≤D
n
2
H0 = ES⊥⊥S′
" D
X
d=0
1
d! ⟨S, S′⟩d
#
,
(86)
where S and S′ are drawn from Dn, and L≤D
n
is the D-low-degree likelihood ratio.
Our SD problem falls under the setting of Lemma 2. Speciﬁcally, let K ∼Unif [Kk,m,n],
and deﬁne ˜S to be an n × n matrix such that [˜S]ij = λ, if i, j ∈K, and [˜S]ij = 0, otherwise.
Also, we deﬁne S as the vectorized version of ˜S. Then, it is clear that our SD problem cast
as the detection problem in Lemma 2, and thus,
L≤D
n
2 = ES⊥⊥S′
" D
X
d=0
1
d! ⟨S, S′⟩d
#
(87)
=
D
X
d=0
λ2d
d! E |K ∩K′|d ,
(88)
where we have used the fact that ⟨S, S′⟩= ST S′ = ∥S ⊙S∥1 = λ2 |K ∩K′|, and K′ is an
independent copy of K. Now, recall that K and K′ are decomposed as K = Sm
ℓ=1 Sℓ× Tℓand
K′ = Sm
ℓ=1 S′
ℓ× T′
ℓ. Thus, we note that the intersection of K and K′ can be rewritten as
|K ∩K′| =
m
X
ℓ1=1
m
X
ℓ2=1
|(Sℓ1 ∩S′
ℓ2) × (Tℓ1 ∩T′
ℓ2)|
(89)
=
m
X
ℓ1=1
m
X
ℓ2=1
|(Sℓ1 ∩S′
ℓ2)| · |(Tℓ1 ∩T′
ℓ2)|.
(90)
For each ℓ1, ℓ2 ∈[m], deﬁne Zℓ1,ℓ2 ≜|(Sℓ1 ∩S′
ℓ2)| and Rℓ1,ℓ2 ≜|(Tℓ1 ∩T′
ℓ2)|. Recall from
the previous subsection that the sequence of random variables {Zℓ1,ℓ2}ℓ1,ℓ2 are statistically
independent of the sequence {Rℓ1,ℓ2}ℓ1,ℓ2, and that Zℓ1,ℓ2 ∼Hypergeometric(n, k, k) and
Rℓ1,ℓ2 ∼Hypergeometric(n, k, k), for each ℓ1, ℓ2 ∈[m], for any ℓ1, ℓ2 ∈[m]. Furthermore,
{Zℓ1,ℓ2}ℓ1,ℓ2 (and similarly {Rℓ1,ℓ2}ℓ1,ℓ2) are negatively associated. Finally, recall that both
Zℓ1,ℓ2 and Rℓ1,ℓ2 are stochastically dominated by Binomial(k, k/n). Thus, using [BT10] (see
also [Ahl22, Theorem 1]), we have
E |K ∩K′|d ≤B2
d max
(
m2k4
n2 ,
m2k4
n2
d)
,
(91)
23

## Page 24

where Bd is the dth Bell number. Thus,
L≤D
n
2 ≤1 +
D
X
d=1
λ2d
d! B2
d max
(
m2k4
n2 ,
m2k4
n2
d)
≜1 +
D
X
d=1
Td.
(92)
If m2k4
n2
< 1, then it is clear that for PD
d=1 Td = O(1), it suﬃces that λ < 1. On the other
hand, if m2k4
n2
> 1, then consider the ratio between successive terms:
Td+1
Td
=
B2
d+1
(d + 1)B2
d
λ2m2 k4
n2.
(93)
Thus if λ is small enough, namely if
mk2λ
n
≤
√
d + 1
√
2
Bd
Bd+1
,
(94)
then Td+1
Td
≤1/2, for all 1 ≤d ≤D. In this case, by comparing with a geometric sum, we
may bound
L≤D
n
2 ≤O(1). This concludes the proof.
To show that the analysis above is tight, note that
L≤D
n
2
D
X
d=0
λ2d
d! E |K ∩K′|d
(95)
≥λ2E |K ∩K′|
(96)
= λ2m2 k4
n2.
(97)
Thus, if λ is large enough, namely if λ = ω(n/(mk2)), then
L≤D
n
2 = ω(1).
4.4
ML Estimator Derivation
The derivation below applies for both the case where K ∈Kk,m,n and the consecutive case
where K ∈Kcon
k,m,n. Let PH1|K(X|K) denote the conditional distribution of X given K. Recall
that the ML estimate of K is given by
ˆKML(X) = arg max
K∈Kk,m,n
log PH1|K(X|K).
(98)
Given K, the distribution of X under H1 is given by,
log PH1|K(X|K) = −n2
2 log(2πe) −1
2
X
(i,j)∈K
(Xij −λ)2 −1
2
X
(i,j)̸∈K
X2
ij
(99)
= −n2
2 log(2πe) + λ2mk2 −1
2
X
(i,j)∈[n]2
X2
ij + λ
2
X
(i,j)∈K
Xij.
(100)
24

## Page 25

Noticing that only the last term at the r.h.s. of (100) depends on K, the ML estimator in
(98) boils down to
ˆKML(X) = arg max
K∈Kk,m,n
X
(i,j)∈K
Xij.
(101)
For the consecutive model, the ML estimator is given by (101), but with Kk,m,n replaced by
Kcon
k,m,n. This problem maximizes the sum of entries among all m principal submatrices of
size k × k of X.
4.5
Proof of Theorem 5
4.5.1
Exact recovery using the ML estimator
In this subsection, we analyze the ML estimator. Recall that,
ˆKML(X) = arg max
K∈Kk,m,n
S(K),
(102)
where S(K) ≜P
(i,j)∈K Xij. We next prove the conditions for which ˆKML = K⋆, with high
probability, where K⋆are the m planted submatrices. To prove the theorem, it suﬃces to
show that S(K⋆) > S(K), for all feasible K with K ̸= K⋆. Let D(K) ≜S(K⋆) −S(K). Note
that
D(K) =
X
(i,j)∈K⋆
Xij −
X
(i,j)∈K
Xij
(103)
=
X
(i,j)∈K⋆
EXij −
X
(i,j)∈K
EXij +
X
(i,j)∈K⋆
[Xij −EXij] −
X
(i,j)∈K
[Xij −EXij]
(104)
= λ · (mk2 −|K⋆∩K|) +
X
(i,j)∈K⋆\K
[Xij −λ] −
X
(i,j)∈K\K⋆
Xij
(105)
= λ · (mk2 −|K⋆∩K|) + W1(K) + W2(K),
(106)
where W1(K) ≜P
(i,j)∈K⋆\K[Xij −λ] and W2(K) ≜−P
(i,j)∈K\K⋆Xij. Because |K| = |K⋆| =
mk2, we have |K⋆\ K| = |K \ K⋆| = mk2 −|K⋆∩K|. Thus, both W1(K) and W2(K) are
composed of the sum of mk2 −|K⋆∩K| i.i.d. centered Gaussian random variables with unit
variance. Accordingly, for i = 1, 2, and each ﬁxed K,
P

Wi(K) ≤−λ
2(mk2 −|K⋆∩K|)

≤1
2 exp

−1
2λ2(mk2 −|K⋆∩K|)

,
(107)
and therefore, by the union bound and (106),
P (D(K) ≤0) ≤exp

−1
2λ2(mk2 −|K⋆∩K|)

.
(108)
Using (108) and the union bound once again, we get
P

ˆKML(X) ̸= K⋆
= P
" [
K̸=K⋆
D(K) ≤0
#
(109)
25

## Page 26

≤
X
K̸=K⋆
P (D(K) ≤0)
(110)
≤
X
K̸=K⋆
exp

−1
2λ2(mk2 −|K⋆∩K|)

(111)
=
mk2−k
X
ℓ=0
K ∈Kcon
k,m,n : |K⋆∩K| = ℓ
 e−1
2 λ2(mk2−ℓ),
(112)
where the last equality follows from the fact that since K⋆, K ∈Kcon
k,m,n and K⋆∩K ̸= ∅, we
must have that |K⋆∩K| ≤mk2 −k. It can be shown that
K ∈Kcon
k,m,n : |K⋆∩K| = ℓ
 ≤
C (mk2−ℓ)2
k2
n
C′(mk2−ℓ)
k
, from some C, C′ > 0, see, e.g., [CX16, Lemma 7]. Then,
P

ˆKML(X) ̸= K⋆
≤C
mk2−k
X
ℓ=0
(mk2 −ℓ)2
k2
n
C′(mk2−ℓ)
k
e−1
2λ2(mk2−ℓ)
(113)
= C
mk2
X
ℓ=k
ℓ2
k2n
C′ℓ
k e−1
2λ2ℓ
(114)
≤C
mk2
X
ℓ=k
n4n
C′ℓ
k e−1
2 λ2ℓ
(115)
= Cn4
mk2
X
ℓ=k
e
C′ℓ
k log n−1
2λ2ℓ
(116)
= Cn4
mk2
X
ℓ=k
e
−ℓ·

1
2 λ2−C′
k log n

(117)
(a)
≤Cn4
mk2
X
ℓ=k
e−8ℓ
k log n
(118)
≤Cn4mk2
n8
= C mk2
n4 ,
(119)
where in (a) we have used the fact that λ2 > (2C′+16) log n
k
. Thus, we get that P(ˆKML(X) ̸= K⋆)
converges to zero, as n →∞.
4.5.2
Exact recovery using the peeling estimator
We analyze the ﬁrst step of the peeling algorithm (which boils down to the ML estimator
for a single planted submatrix), and the strategy to bound each of the other sequential steps
is exactly the same. Recall that,
ˆK1(X) = arg max
K∈Kcon
k,1,n
S(K),
(120)
where S(K) ≜P
(i,j)∈K Xij. We next prove the conditions for which ˆK1(X) = K⋆
ℓ, with high
probability, for some ℓ∈[m], where K⋆= ∪m
ℓ=1K⋆
ℓare the m planted submatrices. To prove
26

## Page 27

the theorem it suﬃces to show that S(K) > maxℓ∈[m] S(K⋆
ℓ) is asymptotically small, for all
feasible K with K ̸= K⋆
ℓ, for ℓ∈[m]. Let Dℓ(K) ≜S(K⋆
ℓ) −S(K). Note that
Dℓ(K) =
X
(i,j)∈K⋆
ℓ
Xij −
X
(i,j)∈K
Xij
(121)
=
X
(i,j)∈K⋆
ℓ
EXij −
X
(i,j)∈K
EXij +
X
(i,j)∈K⋆
ℓ
[Xij −EXij] −
X
(i,j)∈K
[Xij −EXij]
(122)
= λ · (k2 −|K⋆∩K|) +
X
(i,j)∈K⋆
ℓ\K
[Xij −λ] −
X
(i,j)∈K\K⋆
ℓ
[Xij −EXij]
(123)
= λ · (k2 −|K⋆∩K|) + W1(K) + W2(K),
(124)
where W1(K) ≜P
(i,j)∈K⋆
ℓ\K[Xij −λ] and W2(K) ≜−P
(i,j)∈K\K⋆
ℓ[Xij −EXij]. Because |K| =
|K⋆
ℓ| = k2, we have |K⋆
ℓ\ K| = |K \ K⋆
ℓ| = k2 −|K⋆
ℓ∩K|. Thus, both W1(K) and W2(K)
are composed of sum of k2 −|K⋆
ℓ∩K| i.i.d. centered Gaussian random variables with unit
variance. Accordingly, for i = 1, 2, and each ﬁxed K,
P

Wi(K) ≤−λ
2(k2 −|K⋆∩K|)

≤1
2 exp

−λ2
8
(k2 −|K⋆∩K|)2
k2 −|K⋆
ℓ∩K|

.
(125)
Therefore, by the union bound and (124),
P (Dℓ(K) ≤0) ≤exp

−λ2
8
(k2 −|K⋆∩K|)2
k2 −|K⋆
ℓ∩K|

.
(126)
Note that due to the separation assumption, it must be the case that either |K⋆∩K| =
|K⋆
j ∩K| ̸= 0, for some j ∈[m], or |K⋆∩K| = 0. In the later case, we have
P (Dℓ(K) ≤0) ≤exp

−λ2k2
8

,
(127)
while in the former the exists a unique j ∈[m], such that,
min
ℓ∈[m] P (Dℓ(K) ≤0) ≤min
ℓ∈[m] exp

−λ2
8
(k2 −|K⋆
j ∩K|)2
k2 −|K⋆
ℓ∩K|

(128)
≤exp

−λ2
8 (k2 −|K⋆
j ∩K|)

(129)
≤exp

−λ2k
8

,
(130)
where the third inequity is since K⋆
j, K ∈Kcon
k,1,n and K⋆
j ∩K ̸= ∅, we must have that |K⋆
j ∩K| ≤
k2 −k. Accordingly, using (126) and the union bound once again, we get
P

ˆK1(X) ̸= K⋆
ℓfor some ℓ∈[m]

= P


[
K̸=(K⋆
1,...,K⋆m)

S(K) > max
ℓ∈[m] S(K⋆
ℓ)


(131)
27

## Page 28

= P


[
K̸=(K⋆
1,...,K⋆m)
{D1(K) ≤0, . . . , Dm(K) ≤0}


(132)
≤
X
K̸=(K⋆
1,...,K⋆m)
min
ℓ∈[m] P (Dℓ(K) ≤0)
(133)
≤
X
K̸=(K⋆
1,...,K⋆m)
exp

−λ2k
8

(134)
≤n2e−1
8λ2k,
(135)
where the last inequality is because |Kcon
k,1,n| ≤n2. Thus, we see that if λ2 > (24+ǫ) log n
k
, then
P

ˆK1(X) ̸= K⋆
ℓfor some ℓ∈[m]

≤n−(1+ǫ/8). Using the same steps above, it is clear that,
P(ˆKℓ(X) ̸= K⋆
ℓ) ≤n−(1+ǫ/8), for any 2 ≤ℓ≤m, provided that λ2 > (24+ǫ) log n
k
. Thus,
P
h
ˆKpeel ̸= K⋆i
= P
" m
[
ℓ=1
ˆKℓ̸= K⋆
ℓ
#
≤
m
n(1+ǫ/8) = n−ǫ/8,
(136)
which converges to zero as n →∞.
4.5.3
Correlated recovery using the peeling estimator
Our analysis for correlated recovery relies on standard arguments as in [BMV+18, WX20].
Recall the peeling estimator in (20). Denote the planted submatrices by K⋆= Sm
i=1 T⋆
i ×S⋆
i ∈
Kcon
k,m,n. We let K⋆
i ≜T⋆
i × S⋆
i , for i ∈[m], and ﬁx ǫ > 0. Let us analyze the ﬁrst step of the
algorithm, i.e., ˆK1(X(1)) = ˆK1(X). Recall that
ˆK1(X(1)) = arg max
K∈Kcon
k,1,n
S1(K),
(137)
where we deﬁne Sℓ(K) ≜P
(i,j)∈K X(ℓ)
ij , for K ∈Kcon
k,1,n and ℓ∈[m]. Under the planted model,
S1(K) ∼N (λ⟨K, K⋆⟩, k2). Hence, the distribution of S1(K) depends on the size of the overlap
of K with K⋆. To prove that reconstruction is possible, we compute in the planted model
the probability that S1(K) > maxℓ∈[m] S1(K⋆
ℓ), given that K has overlap ⟨K, K⋆⟩= ω with the
planted partition, and argue that this probability tends to zero whenever the overlap is small
enough. For each ℓ∈[m], note that S1(K⋆
ℓ) ∼N (λk2, k2), and thus Hoeﬀding’s inequality
implies that S1(K⋆
ℓ) > λk2 −
p
2k2 log n, with probability at least 1 −O(n−1). Taking the
union bound over every K with overlap at most ω gives
P

max
⟨K,K⋆⟩≤ω S1(K) > λk2 −
p
2k2 log n

≤
(138)
= n2 ·
max
⟨K,K⋆⟩≤ω exp


−
h
λ(k2 −⟨K, K⋆⟩) −
p
2k2 log n
i2
2k2



(139)
28

## Page 29

= n2 ·
max
⟨K,K⋆⟩≤ω exp
 
−

λ
√
2k2(k2 −⟨K, K⋆⟩) −
p
log n
2!
(140)
≤exp
 
2 log n −

λ
√
2k2(k2 −ω) −
p
log n
2!
.
(141)
By the assumption that λ > C√log n
k
, with C > 2 +
√
2, it follows that there exists a ﬁxed
constant ǫ > 0 such that (1 −ǫ)λ > C√log n
k
. Hence, setting ω = k2ǫ in the last displayed
equation, we get
P

max
⟨K,K⋆⟩≤ω S1(K) > λk2 −
p
2k2 log n

≤exp

2 log n −
"
λ
√
k2
√
2
(1 −ǫ) −
p
log n
#2
= e−Ω(n),
(142)
and thus, with probability at least 1 −e−Ω(n),
max
⟨K,K⋆⟩≤k2ǫ S1(K) < λk2 −
p
2k2 log n.
(143)
Consequently, we get that the maximum likelihood estimator ˆK1(X(1)) in (20) satisﬁes
⟨ˆK1, K⋆⟩≥k2ǫ with high probability. Finally, the separation assumption implies that there
exist a unique j ∈[m] such that ⟨ˆK1, K⋆⟩= ⟨ˆK1, K⋆
j⟩≥k2ǫ, and ⟨ˆK1, K⋆
ℓ⟩= 0, for ℓ̸= j. Then,
in the second step of the peeling algorithm, we ﬁrst compute X(2)
ij , by setting [X(2)
ij ]ij = −∞,
for any (i, j) ∈ˆK1, and [X(2)
ij ]ij = 0, otherwise. Thus, it is clear that in the second step, ˆK2
cannot be attained by any set that is k-closed to ˆK1; indeed, S2(K) = −∞, for any set K
that is k-closed to ˆK1. Therefore, for the relevant sets in the maximization in ˆK2, we again
have S2(K) ∼N (λ⟨K, K⋆⟩, k2). Accordingly, repeating the exact same arguments above, we
obtain that ⟨ˆK2, K⋆
j⟩≥k2ǫ with high probability, for some j ∈[m]. In the same way, we get
that ⟨ˆKℓ, K⋆
ℓ⟩≥k2ǫ with high probability, for any ℓ∈[m]. The union bound, then implies
that
P
" m
[
ℓ=1
n
⟨ˆKℓ, K⋆
ℓ⟩< k2ǫ
o#
≤m · P
h
⟨ˆKℓ, K⋆
ℓ⟩< k2ǫ
i
≤me−Ω(n) = o(1).
(144)
Thus, ⟨ˆKpeel, K⋆⟩≥mk2ǫ with high probability, namely, ˆKpeel achieves correlated recovery.
4.6
Proof of Theorem 6
4.6.1
Exact recovery
We use an information theoretical argument via Fano’s inequality.
Recall that Kcon
k,m,n is
the set of possible planted submatrices. Let ¯Kk,m,n be a subset of Kcon
k,m,n, which will be
speciﬁed later on. Let ¯PX,K⋆denote the joint distribution of the underlying location of the
29

## Page 30

planted submatrices K⋆and X, when K⋆is drawn uniformly at random from ¯Kk,m,n, and X
is generated according to Deﬁnition 3. Let I(X; K⋆) denote the mutual information between
X and K⋆. Then, Fano’s inequality implies that,
inf
ˆK
sup
K⋆∈¯Kk,m,n
P
h
ˆK ̸= K⋆i
≥inf
ˆK
¯P
h
ˆK ̸= K⋆i
≥1 −I(X; K⋆) + 1
log | ¯Kk,m,n| .
(145)
We construct ¯Kk,m,n as follows. Let M ≜α · m, where α ∈N will be speciﬁed later on, and
¯Kk,m,n = {Kℓ}M
ℓ=0, where:
1. The
base
submatrix
K0
is
deﬁned
as
K0
=
Sm
ℓ=1 S0
ℓ× T0
ℓ,
with
S0
ℓ
=
{(ℓ−1) · (k + α) + 1, . . . , (ℓ−1) · (k + α) + k} and T0
ℓ= [k], for ℓ∈[m]. Namely,
every pair of consecutive matrices among the m matrices in K0 are α columns far
apart.
2. We let K(j−1)α+i, for j = 1, 2, . . . , m and i = 1, 2, . . . , α, to be deﬁned the same as K0
but with Sj−1 shifted i columns to the right.
Let ¯Pi denote the conditional distribution of X given K⋆= Ki. Note that,
I(X; K⋆) = dKL(¯PX,K⋆||¯PX¯PK⋆)
(146)
= EK⋆
dKL(¯PX|K⋆||¯PX)

(147)
=
1
M + 1
M
X
i=0
dKL(¯Pi||¯PX)
(148)
≤
1
(M + 1)2
M
X
i,j=0
dKL(¯Pi||¯Pj),
(149)
where the inequality follows from the fact that ¯PX =
1
M+1
PM
j=0 ¯Pj, and the convexity of KL
divergence. Now, since each ¯Pi is a product of n2 Gaussian distributions, we get
I(X; K⋆) ≤
1
(M + 1)2
M
X
i,j=0
dKL(¯Pi||¯Pj)
(150)
=
1
(M + 1)
M
X
j=0
dKL(¯P0||¯Pj)
(151)
=
2km
(M + 1) [dKL (N (λ, 1)||N (0, 1)) + dKL (N (0, 1)||N (λ, 1))]
α
X
j=0
j
(152)
=
2km
(M + 1)
α(1 + α)
2
[dKL (N (λ, 1)||N (0, 1)) + dKL (N (0, 1)||N (λ, 1))]
(153)
≤2k(1 + α)
2
[dKL (N (λ, 1)||N (0, 1)) + dKL (N (0, 1)||N (λ, 1))]
(154)
= (1 + α)kλ2.
(155)
30

## Page 31

Thus, substituting the last inequality in (145), and using the fact that | ¯Kk,m,n| = 1 + M, we
get that inf ˆK supK⋆∈¯Kk,m,n P
h
ˆK ̸= K⋆i
> 1/2, if
λ2 <
1
2 log(1 + M) −1
(1 + α)k
=
1
2 log (1 + αm) −1
(1 + α)k
.
(156)
Finally, it is clear that there exists α0 ∈N, such that for any α > α0, the minimax error
probability is at least half, if λ2 < C/k, for some constant C > 0, which concludes the proof.
4.6.2
Correlated recovery
The correlated recovery lower bound follows almost directly from the same arguments as
in, e.g., [WX20, Subsection 3.1.3]. For completeness, we present here the main ideas in
the proof.
Note that the observations can be written as X = λM + W, W is an n × n
i.i.d. matrix with zero mean and unit variance Gaussian entries, and M is an n × n binary
matrix such that Mij = 1 if (i, j) ∈K, and Mij = 0, otherwise, and K is the planted
set. Deﬁne A = βλM + W, where β ∈[0, 1]. The minimum mean-squared error estimator
(MMSE) of M given A is ˆMMMSE = E [M|A], and the rescaled minimum mean-squared error is
MMSE(β) =
1
mk2E ∥M −E [M|A]∥2
F. Note that under the conditions of Theorem 6, we proved
in Theorem 2 that χ2(P||Q) < C, from some constant C > 0. Jensen’s inequality implies
that the KL divergence between P and Q is also bounded, indeed,
dKL(P||Q) ≤log EP
P
Q ≤log C.
(157)
The main idea in the proof is to show that bounded KL divergence implies that for all
β ∈[0, 1], the MMSE tends to that of the trivial estimator ˆM = 0, i.e.,
lim
n→∞MMSE(β) = lim
n→∞
1
mk2E ∥M∥2
F = λ2.
(158)
Expanding the MMSE in the left-hand-side of (158), we get
lim
n→∞
1
mk2E

−2 ⟨M, E [M|A]⟩+ ∥E [M|A]∥2
F

= 0,
(159)
which by the tower property of conditional expectation implies that
lim
n→∞
1
mk2E ∥E [M|A]∥2
F = 0.
(160)
Thus, the optimal estimator converges to the trivial one. To prove (158), a straightforward
calculation shows that the mutual information between A and M is given by I(β) = I(M; A) =
−dKL(P||Q) + β
4E ∥M∥2
F. Thus, under the conditions of Theorem 6,
lim
n→∞
1
mk2I(β) = βλ2
4 .
(161)
Then, using the above and the I-MMSE formula [GSV05] it can be shown that (158) holds
true (see, [WX20, eqns. (13)–(15)]). Next, for any estimator ˆK of the planted set, we can
31

## Page 32

deﬁne an estimator for M by ˆMij = 1 if (i, j) ∈ˆK, and ˆMij = 0, otherwise. Then, using the
Cauchy-Schwarz inequality, we have
E⟨M, ˆM⟩= E⟨E [M|A] , ˆM⟩
(162)
≤E
h
∥E [M|A]∥F || ˆM||F
i
(163)
≤
q
E ∥E [M|A]∥2
Fλ
√
mk2 = o(mk2),
(164)
where the last transition follows from (160). Thus, (164) implies that for any estimator ˆK,
we have E⟨K, ˆK⟩= o(mk2), and thus correlated recovery of K is impossible.
5
Conclusions and future work
In this paper, we study the computational and statistical boundaries of the submatrix and
consecutive submatrix detection and recovery problems. For both models, we derive asymp-
totically tight lower and upper bounds on the thresholds for detection and recovery. To that
end, for each problem, we propose statistically optimal and eﬃcient algorithms for detec-
tion and recovery and analyze their performance. Our statistical lower bounds are based on
classical techniques from information theory. Finally, we use the framework of low-degree
polynomials to provide evidence of the statistical-computational gap we observed in the
submatrix detection problem.
There are several exciting directions for future work. First, it would be interesting to
generalize our results to any pair of distributions P and Q. While our information-theoretic
lower bounds hold for general distributions, it is left to construct and analyze algorithms for
this case, as well as to derive computational lower bounds. In our paper, we assume that
the elements inside the planted submatrices are i.i.d., however, it is of practical interest to
generalize this assumption and consider the case of dependent entries, e.g., Gaussians with
a general covariance matrix. For example, this is the typical statistical model of cryo-EM
data [BBS20]. Finally, it will be interesting to prove a computational lower bound for the
submatrix recovery problem using the recent framework of low-degree polynomials for recov-
ery [SW22], and well as providing other forms of evidence to the statistical computational
gaps for the submatrix detection problem with a growing number of planted submatrices,
e.g., using average-case reductions (see, for example, [BBH18]).
References
[ACCD10] Ery Arias-Castro, Emmanuel Cand`es, and Arnaud Durand.
Detection of an
anomalous cluster in a network. Annals of Statistics, 39, Jan. 2010.
[ACV14] Ery Arias-Castro and Nicolas Verzelen. Community detection in dense random
networks. The Annals of Statistics, 42(3):940–969, 2014.
[Ahl22] Thomas D. Ahle. Sharp and simple bounds for the raw moments of the binomial
and Poisson distributions. Statistics & Probability Letters, 182:109306, 2022.
32

## Page 33

[BBH18] Matthew Brennan, Guy Bresler, and Wasim Huleihel. Reducibility and compu-
tational lower bounds for problems with planted sparse structure. In Proceedings
of the 31st Conference On Learning Theory, volume 75, pages 48–166, 06–09 Jul
2018.
[BBH19] Matthew Brennan, Guy Bresler, and Wasim Huleihel. Universality of computa-
tional lower bounds for submatrix detection. In Proceedings of the Thirty-Second
Conference on Learning Theory, volume 99, pages 417–468, 25–28 Jun 2019.
[BBL+18] Tamir Bendory, Nicolas Boumal, William Leeb, Eitan Levin, and Amit Singer.
Toward single particle reconstruction without particle picking: Breaking the de-
tection limit. arXiv preprint arXiv:1810.00226, 2018.
[BBLS18] Nicolas Boumal, Tamir Bendory, Roy R Lederman, and Amit Singer. Heteroge-
neous multireference alignment: A single pass approach. In 2018 52nd Annual
Conference on Information Sciences and Systems (CISS), pages 1–6. IEEE, 2018.
[BBS20] Tamir Bendory, Alberto Bartesaghi, and Amit Singer.
Single-particle cryo-
electron microscopy: Mathematical theory, computational challenges, and op-
portunities. IEEE signal processing magazine, 37(2):58–76, 2020.
[BDN17] Shankar Bhamidi, Partha Dey, and Andrew Nobel. Energy landscape for large
average submatrix detection problems in gaussian random matrices. Probability
Theory and Related Fields, 168, 08 2017.
[BHK+16] B. Barak, S. B. Hopkins, J. Kelner, P. Kothari, A. Moitra, and A. Potechin. A
nearly tight sum-of-squares lower bound for the planted clique problem. In 2016
IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS),
pages 428–437, 2016.
[BI13] Cristina Butucea and Yuri I Ingster. Detection of a sparse submatrix of a high-
dimensional noisy matrix. Bernoulli, 19(5B):2652–2688, 2013.
[BKR+11] Sivaraman Balakrishnan, Mladen Kolar, Alessandro Rinaldo, Aarti Singh, and
Larry Wasserman. Statistical and computational tradeoﬀs in biclustering. In
NIPS 2011 workshop on computational trade-oﬀs in statistical learning, volume 4,
2011.
[BKW20] Afonso S. Bandeira, Dmitriy Kunisky, and Alexander S. Wein. Computational
Hardness of Certifying Bounds on Constrained PCA Problems. In 11th Inno-
vations in Theoretical Computer Science Conference (ITCS 2020), volume 151,
pages 78:1–78:29, 2020.
[BMR+19] Tristan Bepler, Andrew Morin, Micah Rapp, Julia Brasch, Lawrence Shapiro,
Alex J Noble, and Bonnie Berger.
Positive-unlabeled convolutional neural
networks for particle picking in cryo-electron micrographs.
Nature methods,
16(11):1153–1160, 2019.
33

## Page 34

[BMS15] Xiao-Chen Bai, Greg McMullan, and Sjors HW Scheres. How cryo-EM is revo-
lutionizing structural biology. Trends in biochemical sciences, 40(1):49–57, 2015.
[BMS22] Tamir Bendory, Oscar Mickelin, and Amit Singer. Sparse multi-reference align-
ment:
Sample complexity and computational hardness.
In ICASSP 2022-
2022 IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP), pages 8977–8981. IEEE, 2022.
[BMV+18] Jess Banks, Cristopher Moore, Roman Vershynin, Nicolas Verzelen, and Jiaming
Xu. Information-theoretic bounds and phase transitions in clustering, sparse pca,
and submatrix localization. IEEE Transactions on Information Theory, 2018.
[BPW18] Afonso S Bandeira,
Amelia Perry,
and Alexander S Wein.
Notes on
computational-to-statistical gaps: predictions using statistical physics. Portu-
galiae Mathematica, 75(2):159–186, 2018.
[BT10] Daniel Berend and Tamir Tassa. Eﬃcient bounds on bell numbers and on mo-
ments of sums of random variables. Probability and Mathematical Statistics, 30,
01 2010.
[CC18] Utkan Onur Candogan and Venkat Chandrasekaran. Finding planted subgraphs
with few eigenvalues using the schur–horn relaxation. SIAM Journal on Opti-
mization, 28(1):735–759, 2018.
[CHK+20] Yeshwanth Cherapanamjeri, Samuel B. Hopkins, Tarun Kathuria, Prasad
Raghavendra, and Nilesh Tripuraneni.
Algorithms for heavy-tailed statistics:
Regression, covariance estimation, and beyond. In Proceedings of the 52nd An-
nual ACM SIGACT Symposium on Theory of Computing, STOC 2020, page
601–609, 2020.
[CLR17] Tony Cai, Tengyuan Liang, and Alexander Rakhlin. Computational and statis-
tical boundaries for submatrix localization in a large noisy matrix. Annals of
Statistics, 45(4):1403–1430, 08 2017.
[CX16] Yudong Chen and Jiaming Xu. Statistical-computational tradeoﬀs in planted
problems and submatrix localization with a growing number of clusters and sub-
matrices. Journal of Machine Learning Research, 17(27):1–57, 2016.
[ELS20] Amitay Eldar, Boris Landa, and Yoel Shkolnisky. KLT picker: Particle picking
using data-driven optimal templates. Journal of structural biology, 210(2):107473,
2020.
[GJW20] David Gamarnik, Aukosh Jagannath, and Alexander S. Wein. Low-degree hard-
ness of random optimization problems. In 2020 IEEE 61th Annual Symposium
on Foundations of Computer Science (FOCS), page 324–356, 2020.
[GSV05] Dongning Guo, S. Shamai, and S. Verdu. Mutual information and minimum
mean-square error in Gaussian channels.
IEEE Transactions on Information
Theory, 51(4):1261–1282, 2005.
34

## Page 35

[HAS18] Ayelet Heimowitz, Joakim And´en, and Amit Singer. APPLE picker: Automatic
particle picking, a low-eﬀort cryo-EM framework. Journal of structural biology,
204(2):215–227, 2018.
[HB18] Samuel Hopkins B. Statistical Inference and the Sum of Squares Method. PhD
thesis, Cornell University, 2018.
[Hen95] Richard Henderson. The potential and limitations of neutrons, electrons and X-
rays for atomic resolution microscopy of unstained biological molecules. Quarterly
reviews of biophysics, 28(2):171–193, 1995.
[HKP+17] Samuel B Hopkins, Pravesh K Kothari, Aaron Potechin, Prasad Raghavendra,
Tselil Schramm, and David Steurer. The power of sum-of-squares for detecting
hidden structures. Proceedings of the ﬁfty-eighth IEEE Foundations of Computer
Science (FOCS), pages 720–731, 2017.
[HS17] S. B. Hopkins and D. Steurer. Eﬃcient Bayesian estimation from few samples:
Community detection and related problems. In 2017 IEEE 58th Annual Sympo-
sium on Foundations of Computer Science (FOCS), pages 379–390, 2017.
[Hul22] Wasim Huleihel. Inferring hidden structures in random graphs. IEEE Transac-
tions on Signal and Information Processing over Networks, 8:855–867, 2022.
[HWX15] Bruce Hajek, Yihong Wu, and Jiaming Xu. Computational lower bounds for
community detection on random graphs. In Proceedings of The 28th Conference
on Learning Theory, volume 40, pages 899–928, 03–06 Jul 2015.
[HWX16] Bruce Hajek, Yihong Wu, and Jiaming Xu. Achieving exact cluster recovery
threshold via semideﬁnite programming.
IEEE Transactions on Information
Theory, 62(5):2788–2797, 2016.
[HWX17] Bruce Hajek, Yihong Wu, and Jiaming Xu. Information limits for recovering a
hidden community. IEEE Transactions on Information Theory, 63(8):4729–4745,
2017.
[KB22] Shay Kreymer and Tamir Bendory.
Two-dimensional multi-target detection:
An autocorrelation analysis approach. IEEE Transactions on Signal Processing,
70:835–849, 2022.
[KBRS11] Mladen Kolar, Sivaraman Balakrishnan, Alessandro Rinaldo, and Aarti Singh.
Minimax localization of structural information in large noisy matrices. In Ad-
vances in Neural Information Processing Systems, pages 909–917, 2011.
[KSB23] Shay Kreymer, Amit Singer, and Tamir Bendory.
A stochastic approximate
expectation-maximization for structure determination directly from cryo-EM mi-
crographs. arXiv preprint arXiv:2303.02157, 2023.
[Lyu19] Dmitry Lyumkis. Challenges and opportunities in cryo-EM single-particle anal-
ysis. Journal of Biological Chemistry, 294(13):5181–5197, 2019.
35

## Page 36

[Mon15] Andrea Montanari. Finding one community in a sparse graph. Journal of Sta-
tistical Physics, 161(2):273–299, 2015.
[MRZ15] Andrea Montanari, Daniel Reichman, and Ofer Zeitouni. On the limitation of
spectral methods: From the gaussian hidden clique problem to rank-one per-
turbations of gaussian tensors. In Advances in Neural Information Processing
Systems, pages 217–225, 2015.
[MW15] Zongming Ma and Yihong Wu. Computational barriers in minimax submatrix
detection. Annals of Statistics, 43(3):1089–1116, 2015.
[Sin18] Amit Singer. Mathematics for cryo-electron microscopy. In Proceedings of the
International Congress of Mathematicians: Rio de Janeiro 2018, pages 3995–
4014. World Scientiﬁc, 2018.
[SN13] Xing Sun and Andrew Nobel. On the maximal size of large-average and ANOVA-
ﬁt submatrices in a Gaussian random matrix. Bernoulli, 19:275–294, 02 2013.
[SW22] Tselil Schramm and S. Alexander Wein. Computational barriers to estimation
from low-degree polynomials. Annals of Statistics, 50, Sept. 2022.
[SWP+09] Andrey A Shabalin, Victor J Weigman, Charles M Perou, Andrew B Nobel,
et al. Finding large average submatrices in high dimensional data. The Annals
of Applied Statistics, 3(3):985–1012, 2009.
[Tsy08] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer
Publishing Company, Incorporated, 1st edition, 2008.
[VAC15] Nicolas Verzelen and Ery Arias-Castro. Community detection in sparse random
networks. The Annals of Applied Probability, 25(6):3465–3510, 2015.
[Wei18] Alexander Spence Wein. Statistical estimation in the presence of group actions.
PhD thesis, Massachusetts Institute of Technology, 2018.
[WGL+16] Feng Wang, Huichao Gong, Gaochao Liu, Meijing Li, Chuangye Yan, Tian Xia,
Xueming Li, and Jianyang Zeng. DeepPicker: A deep learning approach for fully
automated particle picking in cryo-EM. Journal of structural biology, 195(3):325–
336, 2016.
[WX20] Yihong Wu and Jiaming Xu.
Statistical problems with planted structures:
Information-theoretical and computational limits. In Miguel R. D. Rodrigues
and Yonina C. Eldar, editors, Information-Theoretic Methods in Data Science.
Cambridge University Press, Cambridge, 2020.
36
