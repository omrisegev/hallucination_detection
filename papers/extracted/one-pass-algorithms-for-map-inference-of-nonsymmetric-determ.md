---
source_pdf: papers/One-Pass Algorithms for MAP Inference of Nonsymmetric Determinantal Point Processes.pdf
slug: one-pass-algorithms-for-map-inference-of-nonsymmetric-determ
pages: 20
extracted_on: 2026-08-04
---

# One-Pass Algorithms for MAP Inference of Nonsymmetric Determinantal Point Processes

## Page 1

One-Pass Algorithms for MAP Inference of
Nonsymmetric Determinantal Point Processes
Aravind Reddy 1 Ryan A. Rossi 2 Zhao Song 2 Anup Rao 2 Tung Mai 2 Nedim Lipka 2 Gang Wu 2
Eunyee Koh 2 Nesreen K. Ahmed 3
Abstract
In this paper, we initiate the study of one-pass
algorithms for solving the maximum-a-posteriori
(MAP) inference problem for Non-symmetric De-
terminantal Point Processes (NDPPs). In particu-
lar, we formulate streaming and online versions
of the problem and provide one-pass algorithms
for solving these problems. In our streaming set-
ting, data points arrive in an arbitrary order and
the algorithms are constrained to use a single-pass
over the data as well as sub-linear memory, and
only need to output a valid solution at the end of
the stream. Our online setting has an additional
requirement of maintaining a valid solution at any
point in time. We design new one-pass algorithms
for these problems and show that they perform
comparably to (or even better than) the offline
greedy algorithm while using substantially lower
memory.
1. Introduction
Determinantal Point Processes (DPPs) were first introduced
in the context of quantum mechanics (Macchi, 1975) and
have subsequently been extensively studied with applica-
tions in several areas of pure and applied mathematics like
graph theory, combinatorics, random matrix theory (Hough
et al., 2006; Borodin, 2009), and randomized numerical
linear algebra (Derezinski & Mahoney, 2021). Discrete
DPPs have gained widespread adoption in machine learning
following the seminal work of (Kulesza & Taskar, 2012)
and have also seen a recent explosion of interest in the ma-
chine learning community. For instance, some of the very
recent uses of DPPs include automation of deep neural net-
work design (Nguyen et al., 2021), deep generative models
(Chen & Ahmed, 2021), document and video summariza-
tion (Perez-Beltrachini & Lapata, 2021), image processing
1Northwestern University 2Adobe Research 3Intel Labs.
Proceedings of the 39 th International Conference on Machine
Learning, Baltimore, Maryland, USA, PMLR 162, 2022. Copy-
right 2022 by the author(s).
(Launay et al., 2021), and learning in games (Perez-Nieves
et al., 2021).
A DPP is a probability distribution over subsets of items and
is characterized by some kernel matrix such that the proba-
bility of sampling any particular subset is proportional to the
determinant of the submatrix corresponding to that subset
in the kernel. Until very recently, most prior work on DPPs
focused on the setting where the kernel matrix is symmet-
ric. Due to this constraint, DPPs can only model negative
correlations between items. Recent work has shown that
allowing the kernel matrix to be nonsymmetric can greatly
increase the expressive power of DPPs and allows them to
model compatible sets of items (Gartrell et al., 2019; Brunel,
2018). To differentiate this line of work from prior litera-
ture on symmetric DPPs, the term Nonsymmetric DPPs
(NDPPs) has often been used. Modeling positive correla-
tions can be useful in many practical scenarios. For instance,
an E-commerce company trying to build a product recom-
mendation system would want the system to increase the
probability of suggesting a router if a customer adds a mo-
dem to a shopping cart.
State-of-the-art algorithms for MAP inference of NDPPs
(Gartrell et al., 2021; Anari & Vuong, 2022) require storing
the full data in memory and take multiple passes over the
complete dataset. Therefore, these algorithms take too much
memory to be useful for large-scale data, where the size of
the entire dataset can be much larger than the random-access
memory available. These algorithms are also not practical
in settings where data is generated on the fly, for example,
in E-commerce applications where new items are added to
the store over time.
Technical contributions.
• We give the first problem formulations for streaming
and online versions of MAP Inference of low-rank-
NDPPs. Our formulations provide a good structure
on how to store NDPP models which are so large that
they cannot fit by themselves in the memory (RAM)
of any single machine (this is an extremely important
practical problem due to the massive scale of industry
datasets) i.e. store the C matrix separately and all the

## Page 2

One-Pass Algorithms for MAP Inference of NDPPs
vi and bi as (vi, bi) pairs (the straight-forward way to
store the data would be to store V , C, B separately
(as in the open-source code provided by Gartrell et al.
(2021)).
• In Section 4, we provide our first streaming algorithm,
PARTITION GREEDY, which is a streaming version of
the standard greedy algorithm for submodular maxi-
mization(Nemhauser et al., 1978), and provide bounds
for the approximation ratio, space used, and time taken.
• In Section 5, we provide ONLINE-LSS and ONLINE-
2-NEIGHBOUR, our online algorithms based on local
search with a stash, which are generalizations of the
online local search algorithm for MAP Inference of
symmetric DPPs by (Bhaskara et al., 2020), and pro-
vide bounds for the space used and time taken.
• In Section 6, we provide a hard instance for one-pass
sublinear-space MAP Inference of NDPPs on which
all of our algorithms fail to output solutions with a
bounded approximation ratio. This illustrates that it
might even be impossible to prove approximation fac-
tor guarantees for our algorithms without additional
strong assumptions. The hard instance uses properties
of NDPPs that differ from symmetric DPPs illustrating
some of the divergence between them. We also provide
some additional comparison between MAP Inference
of nonsymmetric DPPs and symmetric DPPs in this
section.
• In Section 7, we evaluate our proposed online NDPP
MAP Inference algorithms on several datasets and
show that they show that they perform comparably
to (or even better than) the offline greedy algorithm
which takes multiple passes over the data and also uses
substantially more memory (linear in number of items).
2. Related Work
Nonsymmetric DPPs.
A special subset of NDPPs called
signed DPPs were the first class of NDPPs to be studied
(Brunel et al., 2017). Gartrell et al. (2019) studied a more
general class of NDPPs and provided learning and MAP
Inference algorithms, and also showed that NDPPs have
additional expressiveness over symmetric DPPs and can bet-
ter model certain problems. This was improved by Gartrell
et al. (2021) in which they provided a new decomposition
which enabled linear time learning and MAP Inference for
NDPPs. More recently, Anari & Vuong (2022) proposed the
first algorithm with a kO(k) approximation factor for MAP
Inference on NDPPs where k is the number of items to be
selected.
DPP MAP Inference.
MAP Inference in DPPs is a very
well studied NP-hard problem (Ko et al., 1995) with nu-
merous applications in machine learning (Gillenwater et al.,
2012; Han et al., 2017; Chen et al., 2018; Han & Gillenwa-
ter, 2020). Since matrix log-determinant is a submodular
function, several offline algorithms for submodular func-
tion maximization (Krause & Golovin, 2014) have been
applied to the problem. For instance, Civril & Magdon-
Ismail (2009) showed that the standard greedy algorithm
of Nemhauser et al. (1978) provides an O(k!) factor ap-
proximation. Nevertheless, even in the case of symmetric
DPPs, the study of streaming and online algorithms is in
a nascent stage. In particular, (Indyk et al., 2019; 2020;
Mahabadi et al., 2020) provided streaming algorithms for
MAP inference of DPPs and (Bhaskara et al., 2020) were
the first to propose online algorithms for MAP inference of
DPPs. Also, (Liu et al., 2021) designed the first streaming
algorithms for the maximum induced cardinality objective
proposed by (Gillenwater et al., 2018). However, there has
not yet been any work other than ours which has focused on
either streaming or online algorithms for NDPPs.
Streaming and Online Algorithms.
Streaming (Alon
et al., 1999; Muthukrishnan, 2005) and online (Karp et al.,
1990; Karp, 1992; Borodin & El-Yaniv, 2005) algorithms
have been extensively studied in theoretical computer sci-
ence. In particular, they have also seen many applications
in machine learning such as reinforcement learning (Shri-
vastava et al., 2021), projected gradient descent (Xu et al.,
2021), training over-parameterized neural networks (Song
et al., 2021a;b), and solving linear programs (Song & Yu,
2021).
3. Preliminaries
3.1. Notation
Throughout the paper, we use uppercase bold letters (A)
to denote matrices and lowercase bold letters (a) to denote
vectors. Letters in normal font (a) will be used for scalars.
For any positive integer n, we use [n] to denote the set
{1, 2, . . . , n}. A matrix M is said to be skew-symmetric if
M = −M ⊤where ⊤is used to represent matrix transposi-
tion.
3.2. Background on DPPs
A DPP is a probability distribution on all subsets of [n]
characterized by a matrix L ∈Rn×n. The probability
of sampling any subset S ⊆[n] i.e. Pr[S] ∝det(LS)
where LS is the submatrix of L obtained by keeping only
the rows and columns corresponding to indices in S. The
normalization constant for this distribution can be computed
efficiently since we know that P
S⊆[n] det(LS) = det(L +
In) (Kulesza & Taskar, 2012, Theorem 2.1). Therefore,

## Page 3

One-Pass Algorithms for MAP Inference of NDPPs
Table 1. Summary of our MAP inference algorithms for NDPPs. We omit O for simplicity. All algorithms use only a single-pass over the
data.
Inference Problem
Algorithm
Update Time
Total Time
Space
Streaming
STREAM-PARTITION (Alg. 1)
N/A
Tdet(k, d)·n
k2+d2
Online
ONLINE-LSS (Alg. 2)
Tdet(k, d)·k log2(∆)
Tdet(k, d)·(nk+k log2(∆))
k2+d2+d logα(∆)
ONLINE 2-NEIGH (Alg. 3)
Tdet(k, d)·k2 log3(∆)
Tdet(k, d)·
 nk2+k2 log3(∆)

k2+d2+d logα(∆)
ONLINE-GREEDY (Alg. 4)
Tdet(k, d)·k
Tdet(k, d)·nk
k2+d2
Pr[S] =
det(LS)
det(L+In). For the DPP corresponding to L to be
a valid probability distribution, we need det(LS) ≥0 for
all S ⊆[n] since Pr[S] ≥0 for all S ⊆[n]. Matrices that
satisfy this property are known as P0-matrices (Fiedler &
Pták, 1966). For any symmetric matrix L, det(LS) ≥0 for
all S ⊆[n] if and only if L is positive semi-definite (PSD)
i.e. xT Lx ≥0 for all x ∈Rn. Therefore, all symmetric
matrices which correspond to valid DPPs are PSD. But there
are P0-matrices that are not necessarily symmetric (or even
positive semi-definite). For example, L =
 1
1
−1
1

is a
nonsymmetric P0 matrix.
Any matrix L can be uniquely written as the sum of a sym-
metric and skew-symmetric matrix: L = (L + L⊤)/2 +
(L −L⊤)/2. For the DPP characterized by L, the sym-
metric part of the decomposition can be thought of as en-
coding negative correlations between items and the skew-
symmetric part as encoding positive correlations. Gartrell
et al. (2019) proposed a decomposition that covers the set
of all nonsymmetric PSD matrices (a subset of P0 matri-
ces) which allowed them to provide a cubic time algorithm
(in the ground set size) for NDPP learning. This decom-
position is L = V ⊤V + (BC⊤−CB⊤). Gartrell et al.
(2021) provided more efficient (linear time) algorithms for
learning and MAP inference using a new decomposition
L = V ⊤V + B⊤CB. Although both these decomposi-
tions only cover a subset of P0 matrices, it turns out that
they are quite useful for modeling real-world instances and
provide improved results when compared to (symmetric)
DPPs.
For the decomposition L = V ⊤V + B⊤CB, we have
V , B ∈Rd×n, C ∈Rd×d and C is skew-symmetric. Here
we can think of the items having a latent low-dimensional
representation (vi, bi) where vi, bi ∈Rd. Intuitively, a
low-dimensional representation (when compared to n) is
sufficient for representing items because any particular item
only interacts with a small number of other items in real-
world datasets, as evidenced by the fact that the maximum
basket size encountered in real-world data is much smaller
than n.
3.3. Background on Streaming and Online Algorithms
The main difference between streaming and online algo-
rithms are their parameters of interest. In streaming algo-
rithms (Muthukrishnan, 2005), the main focus is on the
solution quality at the end of the stream and memory used
by the algorithm throughout the stream. Instead, in online
algorithms (Karp et al., 1990), the main focus is on the
solution quality at every time step and update time after
seeing a new input. In the streaming and online models we
will define, the online setting is a more restrictive version
of the streaming setting. Therefore, for us, any algorithms
which are valid online algorithms are also valid streaming
algorithms. However, not all streaming algorithms are on-
line algorithms. For instance, our main streaming algorithm
(Algorithm 1) is not a valid online algorithm.
4. Streaming MAP Inference
In this section, we formulate the streaming MAP inference
problem for NDPPs and design an algorithm for this prob-
lem with guarantees on the solution quality, space, and time.
4.1. Streaming MAP Inference Problem
We study the MAP Inference problem in low-rank NDPPs
in the streaming setting where we see columns of a 2d × n
matrix in order (column-arrival model). Given some fixed
skew-symmetric matrix C ∈Rd×d, consider a stream of
2d-dimensional vectors (which can be viewed as pairs of
d-dimensional vectors) arriving in order:
(v1, b2), (v2, b2), . . . , (vn, bn) where vt, bt ∈Rd, ∀t ∈[n]
The main goal in the streaming setting is to output the maxi-
mum likelihood subset S ⊆[n] of cardinality k at the end
of the stream assuming that S is drawn from the NDPP
characterized by L = V ⊤V + B⊤CB i.e.
S = argmax
S⊆[n],|S|=k
det(LS)
(1)
= argmax
S⊆[n],|S|=k
det(V ⊤
S VS + B⊤
S CBS)
For any S ⊆[n], VS ∈Rd×|S| is the matrix whose each col-

## Page 4

One-Pass Algorithms for MAP Inference of NDPPs
Algorithm 1 Streaming Partition Greedy MAP Inference
for low-rank NDPPs
1: Input: Length of the stream n and a stream of data
points {(v1, b1), (v2, b2), . . . , (vn, bn)}
2: Output: A solution set S of cardinality k at the end of
the stream.
3: S0 ←∅, s0 ←∅
4: while new data (vt, bt) arrives in stream at time t do
5:
i ←⌈tk
n ⌉
6:
if f(Si−1 ∪{t}) > f(Si−1 ∪{si}) then
7:
si ←t
8:
if t is a multiple of n
k then
9:
Si ←Si−1 ∪si
10:
si ←∅
11: return Sk
umn corresponds to {vi, i ∈S}. Similarly, BS ∈Rd×|S| is
the matrix whose columns correspond to {bi, i ∈S}. In the
case of symmetric DPPs, this maximization problem in the
non-streaming setting corresponds to MAP Inference in car-
dinality constrained DPPs, also known as k-DPPs (Kulesza
& Taskar, 2011).
Definition 1. Given three matrices V ∈Rd×k, B ∈Rd×k
and C ∈Rd×d, let Tdet(k, d) denote the running time of
computing det(V ⊤V + B⊤CB). We can take Tdet(k, d)
being O(kd2) as a crude estimate.
Note that Tdet(k, d) = 2Tmat(d, k, d) + Tmat(d, d, k) +
Tmat(k, k, k) where Tmat(a, b, c) is the time required to mul-
tiply two matrices of dimensions a × b and b × c. We have
the last Tmat(k, k, k) term because computing the determi-
nant of a k × k matrix can be done (essentially) in the same
time as computing the product of two matrices of dimension
k × k (Aho et al., 1974, Theorem 6.6).
We will now describe a streaming algorithm for MAP in-
ference in NDPPs, which we call the "Streaming Partition
Greedy" algorithm.
4.2. Streaming Partition Greedy
Outline of Algorithm 1: Our algorithm picks the first el-
ement of the solution greedily from the first seen n
k ele-
ments, the second element from the next sequence of n
k
elements and so on. As described in Algorithm 1, let us
use S0, S1, . . . , Sk to denote the solution sets maintained by
the algorithm, where Si represents the solution set of size
i. In particular, we have that Si = Si−1 ∪{si} where si =
arg maxj∈Pi f(S ∪{j}) and Pi denotes the i’th partition
of the data i.e. Pi := { (i−1)·n
k
+ 1, (i−1)·n
k
+ 2, . . . , i·n
k }.
(v1, b1), (v2, b2), ..., (vn/k, bn/k)
|
{z
}
P1
, . . . ,
(2)
(vn−(n/k)+1, bn−(n/k)+1), ..., (vn, bn)
|
{z
}
Pk
Theorem 2. For a random-order arrival stream, if S is
the solution output by Algorithm 1 at the end of the stream
and σmin > 1 where σmin and σmax denote the smallest
and largest singular values of LS among all S ⊆[n] and
|S| ≤2k, then
E[log det(LS)]
log(OPT)
≥
 
1 −
1
σ
(1−1
e )·(2 log σmax−log σmin)
min
!
where LS
=
V ⊤
S VS + B⊤
S CBS
and OPT
=
max
R⊆[n], |R|=k det(V ⊤
R VR + B⊤
RCBR).
We will first give a high-level proof sketch for this theorem
and defer the full proof to Appendix A.
Proof sketch. For a random-order arrival stream, the distri-
bution of any set of consecutive n/k elements is the same
as the distribution of n/k elements picked uniformly at
random (without replacement) from [n]. If the objective
function f is submodular, then this algorithm has an approx-
imation guarantee of (1 −2/e) by (Mirzasoleiman et al.,
2015). But neither det(LS) nor log det(LS) are submod-
ular. Instead, (Gartrell et al., 2021) [Equation 45] showed
that log det(LS) is “close” to submodular when σmin > 1
where this closeness is measured using a parameter known
as “submodularity ratio” (Bian et al., 2017). Using this
parameter, we can prove a guarantee for our algorithm.
■
Theorem 3. For any length-n stream (v1, b1), . . . , (vn, bn)
where (vt, bt) ∈Rd × Rd ∀t ∈[n], the space used is
O(k2 + d2) and the total time taken is O(n · Tdet(k, d))
where Tdet(k, d) is the time taken to compute f(S) =
det(V ⊤
S V + B⊤
S CB) for |S| = k.
Proof. For any particular data-point (vt, bt), Algorithm 1
needs to compute f(Si−1 ∪{t}),
where f(S)
=
det(V ⊤
S V + B⊤
S CBS) and S = Si−1 ∪{t}. This takes
at most O(k2 + d2) space and Tdet(k, d) time. The algo-
rithm also needs to store Si−1, si and f(Si−1 ∪{si}) but
all of these are dominated by O(k2 + d2) space needed to
compute the determinant. All the other comparison and
update steps are also much faster and so the total time is
O(n · Tdet(k, d))
■

## Page 5

One-Pass Algorithms for MAP Inference of NDPPs
Algorithm 2 ONLINE-LSS: Online MAP Inference for low-
rank NDPPs with Stash.
1: Input:
A
stream
of
data
points
{(v1, b1), . . . , (vn, bn)}, and a constant α ≥1
2: Output: A solution set S of cardinality k at the end of
the stream.
3: S, T ←∅
4: while new data point (vt, bt) arrives in stream at time t
do
5:
if |S| < k and f (S ∪{t}) ̸= 0 then
6:
S ←S ∪{t}
7:
else
8:
i ←arg maxj∈S f(S ∪{t} \ {j})
9:
if f (S ∪{t} \ {i}) > α · f(S) then
10:
S ←S ∪{t} \ {i}
11:
T ←T ∪{i}
12:
while
∃
a
∈
S,
b
∈
T
:
f (S ∪{b} \ {a}) > α · f(S) do
13:
S ←S ∪{b} \ {a}
14:
T ←T ∪{a} \ {b}
15: return S
5. Online MAP Inference for NDPPs
We now consider the online MAP inference problem for
NDPPs, which is natural in settings where data is generated
on the fly. In addition to the constraints of the streaming
setting (Section 4.1), our online setting requires us to main-
tain a valid solution at every time step. In this section, we
provide two algorithms for solving this problem.
5.1. Online Local Search with a Stash
Outline of Algorithm 2: On a high-level, our algorithm is
a generalization of the Online-LS algorithm for DPPs from
(Bhaskara et al., 2020). At each time step t ∈[n] (after
t ≥k), our algorithm maintains a candidate solution subset
of indices S of cardinality k from the data seen so far i.e.
S ⊆[t] s.t. |S| = k in a streaming fashion. Additionally,
it also maintains two matrices VS, BS ∈Rd×|S| where the
columns of VS are {vi, i ∈S} and the columns of BS are
{bi, i ∈S}. Whenever the algorithm sees a new data point
(vt, bt), it replaces an existing index from S with the newly
arrived index if doing so increases f(S) at-least by a factor
of α ≥1 where α is a parameter to be chosen (we can
think of α being 2 for understanding the algorithm). Instead
of just deleting the index replaced from S, it is stored in
an auxiliary set T called the “stash" (and also maintains
corresponding matrices VT , BT ), which the algorithm then
uses to perform a local search over to find a locally optimal
solution.
We now define a data-dependent parameter ∆which we will
need to describe the time and space used by Algorithm 2.
Definition 4. Let the first non-zero value of f(S) with |S| =
k that can be achieved in the stream without any swaps be
valnz i.e. till S reaches a size k, any index seen is added to S
if f(S) remains non-zero even after adding it. Let us define
∆:= OPTk
valnz where OPTk = maxS⊆[n],|S|=k det(V ⊤
S VS +
B⊤
S CBS).
Note that ∆is a data-dependent parameter which can poten-
tially be unbounded. This happens in the hard-instance we
will describe in Section 6. However, for practical datasets,
∆doesn’t seem to be too bad (see the experiments section
for a more detailed discussion).
Theorem 5. For any length-n stream (v1, b1), . . . , (vn, bn)
where (vt, bt) ∈Rd × Rd ∀t ∈[n], the worst case up-
date time of Algorithm 2 is O(Tdet(k, d) · k log2(∆)) where
Tdet(k, d) is the time taken to compute f(S) = det(V ⊤
S V +
B⊤
S CB) for |S| = k. Furthermore, the amortized update
time is O(Tdet(k, d) · (k + k log2(∆)
n
)) and the space used
at any time step is at most O(k2 + d2 + d logα(∆)).
Proof. For every iteration of the while loop in line 4: It takes
at most Tdet(k, d) time for checking the first if condition
(lines 5-6). The argmaxj∈S f(S ∪{t} \ {j} step takes at
most k·Tdet(k, d) time. The while loop in line 12 takes time
at most |S|·|T|·Tdet(k, d) for every instance of an increase
in f(S). Note that f(S) can increase at most logα(∆) times
since the value of f(S) cannot exceed OPTk. Therefore,
the update time of Algorithm 2 is at most Tdet(k, d) + k ·
Tdet(k, d)+logα(∆)·(|S| · |T| · Tdet(k, d)) ≤Tdet(k, d)·
 k + 1 + k log2
α(∆)

since |S| ≤k and |T| ≤logα(∆).
Notice that the cardinality of T can increase by 1 only when
the value of f(S) increases at least by a factor of α and so
|T| ≤logα(∆).
During any time step t, the algorithm needs to store
the indices in S, T
and the corresponding matrices
VS, BS, VT , BT . Since |S| ≤k, |T| ≤logα(∆) and it
takes d words to store every vi and bi, we need at most
k + logα(∆) + 2dk + 2d logα(∆) words to store all these
in memory. The space needed to compute det(V ⊤
S VS +
B⊤
S CBS) is at most O(k2 + d2). We compute all such
determinants one after the other in our algorithm. So the
algorithm only needs space for one such computation dur-
ing its run. Therefore, the space required by Algorithm 2 is
O(k2 + d2 + d logα(∆)).
■
5.2. Online 2-neighborhood Local Search Algorithm
with a Stash
Before we describe our algorithm, we will define a
neighborhood of any solution, which will be useful for de-
scribing the local search part of our algorithm.
Definition 6 (Nr(S, T)). For any natural number r ≥1
and any sets S, T we define the r-neighborhood of S with

## Page 6

One-Pass Algorithms for MAP Inference of NDPPs
Algorithm 3 ONLINE-2-NEIGHBOR: Local Search over 2-neighborhoods with Stash for Online NDPP MAP Inference.
1: Input: A stream of data points {(v1, b1), (v2, b2), . . . , (vn, bn)}, and a constant α ≥1
2: Output: A solution set S of cardinality k at the end of the stream.
3: S, T ←∅
4: while new data (vt, bt) arrives in stream at time t do
5:
if |S| < k and f (S ∪{t}) ̸= 0 then
6:
S ←S ∪{t}
7:
else
8:
{i, j} ←arg maxa,b∈S (f(S ∪{t} \ {a}), f(S ∪{t −1, t} \ {a, b}))
9:
fmax ←maxa,b∈S (f(S ∪{t} \ {a}), f(S ∪{t −1, t} \ {a, b}))
10:
if fmax > α · f(S) then
11:
if two items are chosen to be replaced: then
12:
S ←S ∪{t −1, t} \ {i, j}
13:
T ←T ∪{i, j}
14:
else
15:
S ←S ∪{t} \ {i}
16:
T ←T ∪{i}
17:
while ∃a, b ∈S, c, d ∈T : f (S ∪{c, d} \ {a, b}) > α · f(S) do
18:
S ←S ∪{c, d} \ {a, b}
19:
T ←T ∪{a, b} \ {c, d}
20: return S
respect to T
Nr(S, T) := {S′ ⊆S ∪T | |S′| = |S| and |S′ \ S| ≤r}
Outline of Algorithm 3: Similar to Algorithm 2, our new
algorithm also maintains two subsets of indices S and T, and
corresponding data matrices VS, BS, VT , BT . Whenever
the algorithm sees a new data-point (vt, bt), it checks if
the solution quality f(S) can be improved by a factor of
α by replacing any element in S with the newly seen data-
point. Additionally, it also checks if the solution quality
can be made better by including both the points (vt, bt) and
the data-point (vt−1, bt−1). Further, the algorithm tries to
improve the solution quality by performing a local search
on N2(S, T) i.e. the neighborhood of the candidate solution
S using the stash T by replacing at most two elements of
S. There might be interactions captured by pairs of items
which are much stronger than single items in NDPPs (see
example 5 from (Anari & Vuong, 2022)).
Theorem 7. For any length-n stream (v1, b1), . . . , (vn, bn)
where (vt, bt) ∈Rd × Rd ∀t ∈[n], the worst case
update time of Algorithm 3 is O(Tdet(k, d) · k2 log3(∆))
where Tdet(k, d) is the time taken to compute f(S) =
det(V ⊤
S V + B⊤
S CB) for |S| = k. The amortized update
time is O

Tdet(k, d) ·

k2 + k2 log3(∆)
n

and the space
used at any time step is at most O(k2 + d2 + d logα(∆)).
Proof. It
takes
at
most
Tdet(k, d)
time
for
lines
5-6
(same
as
in
LSS).
The
arg maxa,b∈S (f(S ∪{t} \ {a}), f(S ∪{t −1, t} \ {a, b}))
step takes at most k2 · Tdet(k, d) time. The while loop
in line 18 takes time at most |S|2 · |T|2 · Tdet(k, d) for
every instance of an increase in f(S). Similar to LSS,
f(S) can increase at most by a factor of logα(∆) since
the value of f(S) cannot exceed OPTk.
Therefore,
the update time of Algorithm 3 is at most Tdet(k, d) +
k2 · Tdet(k, d) + logα(∆) ·

|S|2 · |T|2 · Tdet(k, d)

≤
Tdet(k, d) ·
 k2 + 1 + k2 log3
α(∆)

since |S| ≤k and
|T| ≤logα(∆).
Although Algorithm 3 executes more number of determi-
nant computations than Algorithm 2, all of them are done
sequentially and only the maximum value among all the
previously computed values in any specific iteration needs
to be stored in memory. Therefore, the space needed is
(nearly) the same for both algorithms.
■
6. Hard Instance for One-Pass MAP Inference
of NDPPs
We will now give a high-level description of a hard instance
for one-pass MAP inference of NDPPs with sub-linear mem-
ory (inspired by (Anari & Vuong, 2022, Example 5)) on
which all of our algorithms output solutions with zero objec-
tive value whereas the optimal solution has non-zero value.
Due to this, we believe it might be impossible to prove any
bounded approximation factor guarantees for our algorithms
without any strong additional assumptions.

## Page 7

One-Pass Algorithms for MAP Inference of NDPPs
Sketch of the hard instance.
Suppose we have a total of
2d items consisting of pairs of complementary items like
modem-router, printer-ink cartridge, pencil-eraser, etc. Let
us use {1, 1c, 2, 2c, . . . , d, dc} to denote them. Any item i
is independent of every item other than its complement ic.
Individually, Pr[{i}] = Pr[{ic}] = 0 . And Pr[{i, ic}] =
x2
i with xi > 0 for all i ∈[d]. Also, we have Pr[{i, j}] = 0
for any i ̸= j. Suppose any of our online algorithms are
given the sequence {1, 2, 3, . . . , d, rc} where r ∈[d] is
some arbitrary item and the algorithm needs to pick 2 items
i.e. k = 2. Then, OPT > 0 whereas all of our online
algorithms (Online LSS, Online 2-neighbor, Online-Greedy)
will fail to output a valid solution. We provide a more
complete description with the instantiations of the matrices
B, C, V in Appendix B.
Comparison between MAP Inference for NDPPs and
symmetric DPPs.
The hard instance described above nec-
essarily uses the skew-symmetric component of the kernel
matrix to form such complementary pairs and this illustrates
some of the divergence between NDPPs and symmetric
DPPs. NDPPs are significantly more general (and complex)
than DPPs and unlike the case for DPPs where the objective
function corresponds to a nice geometric notion i.e. volume,
the objective function for MAP Inference on NDPPs doesn’t
have a corresponding clean notion. This is a core issue be-
cause of which it is unclear how the proofs of approximation
factors for similar algorithms for DPPs would generalize to
NDPPs (the proof techniques for DPPs from Bhaskara et al.
(2020) for Online-LS heavily use the fact that the objective
function is a volume and thus use properties of coresets
developed for related geometric problems).
7. Experiments
We first learn all the matrices B, C, and V by applying the
learning algorithm of (Gartrell et al., 2021) on several real-
world datasets. For example, some datasets consist of carts
of items bought by Amazon customers. Then, we run our
inference algorithms on the learned B, C, and V . More
details about the experiments and the datasets used can be
found in Appendix C.
As a point of comparison, we also use the offline greedy
algorithm from (Gartrell et al., 2021). This algorithm stores
all data in memory and makes k passes over the dataset
and in each round, picks the data point which gives the
maximum marginal gain in solution value. Online-Greedy
(Algorithm 4) is a simple online variant of the greedy algo-
rithm which replaces a point in the current solution set with
the observed point if doing so increases the objective.
First, we want to mention that the performance of our stream-
ing algorithm (Algorithm 1) on the datasets we consider is
(unfortunately) pretty bad (the objective function value is
Algorithm 4 ONLINE-GREEDY: Online Greedy MAP In-
ference for NDPPs
1: Input:
A
stream
of
data
points
{(v1, b1), (v2, b2), . . . , (vn, bn)}
2: Output: A solution set S of cardinality k at the end of
the stream.
3: S ←∅
4: while new data (vt, bt) arrives in stream at time t do
5:
if |S| < k and f (S ∪{t}) ̸= 0 then
6:
S ←S ∪{t}
7:
else
8:
i ←arg maxj∈S f(S ∪{t} \ {j})
9:
if f (S ∪{t} \ {i}) > f(S) then
10:
S ←S ∪{t} \ {i}
11: return S
close to zero in most cases). This is because in real-world
datasets, adjacent pairs of items have a very high likelihood
of occurring together when compared to pairs of items far
from each other. For example, white socks and grey socks
might be adjacent to each other in numbering. And cus-
tomers tend to buy both of them in a basket. Our streaming
algorithm is forced to ignore most such pairs.
Results for a various datasets for our online algorithms are
provided in Figure 1. Surprisingly, the solution quality of
our online algorithms compare favorably with the offline
greedy algorithm while using only a single-pass over the
data, and a tiny fraction of memory. In most cases, Online-
2-neighbor (Algorithm 3) performs better than Online-LSS
(Algorithm 2) which in turn performs better than the online
greedy algorithm (Algorithm 4). Strikingly, our online-2-
neighbor algorithm performs even better than offline greedy
in many cases.
We also perform several experiments comparing the num-
ber of determinant computations (as a system-independent
proxy for time) and the number of swaps (as a measure
of solution consistency) of all our online algorithms. Re-
sults for determinant computations (Figure 2) and swaps
(Figure 3) can be found in Appendix D. We summarize the
main findings here. The number of determinant computa-
tions of Online-LSS is comparable to that of Online Greedy
but the number of swaps performed is significantly smaller.
Online-2-neighbor is the most time-consuming but superior
performing algorithm in terms of solution quality.
Our experimental results in Appendix D demonstrate that
our online algorithms use substantially lower memory than
any offline algorithms. Note that the main memory bottle-
neck for offline inference algorithms (Gartrell et al., 2021;
Anari & Vuong, 2022) is the need to store the entire data-set
in memory. We can consider other factors (like the memory
needed for computing det(k, d) i.e. O(k2 + d2) (which can

## Page 8

One-Pass Algorithms for MAP Inference of NDPPs
0
500 1000 1500 2000 2500 3000 3500 4000
Data points analyzed
0.0e+00
1.0e-07
2.0e-07
3.0e-07
4.0e-07
5.0e-07
6.0e-07
7.0e-07
Solution
UK Retail
Online 2-neighbor
Online LSS
Online Greedy
Offline
0
2000
4000
6000
8000 10000 12000
Data points analyzed
0.0e+00
1.0e-19
2.0e-19
3.0e-19
4.0e-19
5.0e-19
6.0e-19
Solution
MovieLens
Online 2-neighbor
Online LSS
Online Greedy
Offline
20
40
60
80
100
Data points analyzed
0.0e+00
2.0e-09
4.0e-09
6.0e-09
8.0e-09
1.0e-08
1.2e-08
1.4e-08
Solution
Amazon Apparel
Online 2-neighbor
Online LSS
Online Greedy
Offline
0
50
100
150
200
250
300
Data points analyzed
0.0e+00
5.0e-05
1.0e-04
1.5e-04
2.0e-04
2.5e-04
Solution
Amazon 3-category
Online 2-neighbor
Online LSS
Online Greedy
Offline
0
10000
20000
30000
40000
50000
Data points analyzed
0.0e+00
2.0e-09
4.0e-09
6.0e-09
8.0e-09
Solution
instacart
Online 2-neighbor
Online LSS
Online Greedy
Offline
0
50000 100000150000200000250000300000350000
Data points analyzed
0.0e+00
5.0e+07
1.0e+08
1.5e+08
2.0e+08
2.5e+08
Solution
Million Song
Online 2-neighbor
Online LSS
Online Greedy
Offline
0
200
400
600
800
1000
1200
Data points analyzed
0.0e+00
1.0e-07
2.0e-07
3.0e-07
4.0e-07
5.0e-07
6.0e-07
7.0e-07
8.0e-07
Solution
Customer Dashboards
Online 2-neighbor
Online LSS
Online Greedy
Offline
Company Retail
Figure 1. Solution quality i.e. objective function value as a function of the number of data points analyzed for all our online algorithms
and also the offline greedy algorithm. All our online algorithms give comparable (or even better) performance to offline greedy using only
a single pass and a small fraction of the memory.
be re-used every-time) to be essentially free because the
regime of interest is n ≫d ≥k. The memory usage by our
online algorithms is also primarily dominated by the size of
the stash, which is upper bounded by the number of swaps
for which we have plots in Appendix D.2. Similarly, the up-
date times also depend only on the size of the stash (which
are quite small and so we have very fast update times).
We also investigate the performance of our algorithms under
the random stream paradigm, where we consider a random
permutation of some of the datasets used earlier. Results
for the solution quality (Figure 4), number of determinant
computations and swaps (Figure 5) can be found in Ap-
pendix D.3. In this setting, we see that Online-LSS and
Online-2-neighbor have nearly identical performance and
are always better than Online-Greedy in terms of solution
quality and number of swaps.
We study the effect of varying α in Online-LSS (Algo-
rithm 2) for various values of set sizes k in Appendices
D.4 and D.5. We notice that, in general, the solution quality,
number of determinant computations, and the number of
swaps increase as α decreases (Figure 6). We also see that
as k increases, the solution value decreases across all values
of α (Figure 7. This is in accordance with our intuition that
the probability of larger sets should be smaller.
8. Conclusion & Future Directions
In this paper, we formulate and study the streaming and
online MAP inference problems for Nonsymmetric Deter-
minantal Point Processes. To the best of our knowledge,
this is the first work to study these problems in these practi-
cal settings. We design new one-pass algorithms for these
problems, prove theoretical guarantees for them in terms
of space required, time taken, and solution quality for our
algorithms, and empirically show that they perform compa-
rably or sometimes even better than state-of-the-art offline
algorithms while using substantially lower memory.
As we have discussed in the experiments section, the em-
pirical performance of our partition greedy algorithm is

## Page 9

One-Pass Algorithms for MAP Inference of NDPPs
quite bad. The main reason we have chosen to include it in
this paper is because it is the only one for which we have
a provable guarantee on the approximation quality (albeit
with strong assumptions). This also leads us to an important
open direction from our work, i.e. gaining a better theo-
retical understanding of our online algorithms, potentially
by proving approximation bounds going beyond worst-case
analysis (Roughgarden, 2021). For instance, by assuming
that the learned NDPP model satisfies natural assumptions
like perturbation stability (Bilu & Linial, 2012; Makarychev
et al., 2014; Angelidakis et al., 2017). For example, in the
line of prior work (Lang et al., 2018; 2019; 2021a;b) study-
ing MAP inference for Potts models. Another interesting
direction is in providing parallelizable algorithms which use
a small number of passes (greater than one but less than
k) - similar to the k-means∥(read as “k-means parallel”)
algorithm (Bahmani et al., 2012; Makarychev et al., 2020)
We have only studied 1-neighbor and 2-neighbor online
local search algorithms in our paper. Extending them to
arbitrary sizes of subsets (3-neighbor, etc.) is also another
interesting open direction. Understanding at which point
the degree of interactions cease to provide benefits that are
worth the increase in memory/time constraints would be
of interest to the DPP research community. Are pairwise
interactions, as in 2-neighbor, sufficient to characterize most
of the necessary NDPP properties?
Acknowledgments
We would like to thank all the ICML 2022 reviewers of our
paper and also the ICLR 2022 reviewers who reviewed an
earlier version of this work, for their very valuable feedback.
Most of this work was done while AR was an intern with
Adobe Research, San Jose, CA, USA in the summer of 2021.
AR was also supported by NSF CCF-1652491 and NSF
CCF-1955351 during the preparation of this manuscript.
References
Aho, A., Hopcroft, J., and Ullman, J. The Design and
Analysis of Computer Algorithms. Addison-Wesley, 1974.
Alon, N., Matias, Y., and Szegedy, M. The space complexity
of approximating the frequency moments. Journal of
Computer and system sciences, 58(1):137–147, 1999.
Anari, N. and Vuong, T.-D. From Sampling to Optimiza-
tion on Discrete Domains with Applications to Determi-
nant Maximization. In Conference on Learning Theory
(COLT), 2022.
Angelidakis, H., Makarychev, K., and Makarychev, Y. Al-
gorithms for stable and perturbation-resilient problems.
In Proceedings of the 49th Annual ACM SIGACT Sympo-
sium on Theory of Computing, pp. 438–451, 2017.
Bahmani, B., Moseley, B., Vattani, A., Kumar, R., and
Vassilvitskii, S. Scalable k-means++. Proceedings of the
VLDB Endowment, 5(7):622–633, 2012.
Bhaskara, A., Karbasi, A., Lattanzi, S., and Zadimoghad-
dam, M. Online MAP Inference of Determinantal Point
Processes. In Neural Information Processing Systems
(NeurIPS), 2020.
Bian, A. A., Buhmann, J. M., Krause, A., and Tschi-
atschek, S. Guarantees for Greedy Maximization of Non-
submodular Functions with Applications. In International
Conference on Machine Learning (ICML), 2017.
Bilu, Y. and Linial, N. Are stable instances easy? Com-
binatorics, Probability and Computing, 21(5):643–660,
2012.
Borodin, A. Determinantal point processes. In The Oxford
Handbook of Random Matrix Theory. Oxford University
Press, 2009.
Borodin, A. and El-Yaniv, R. Online Computation and
Competitive Analysis. Cambridge University Press, 2005.
Brunel, V.-E. Learning Signed Determinantal Point Pro-
cesses through the Principal Minor Assignment Problem.
In Neural Information Processing Systems (NeurIPS),
2018.
Brunel, V.-E., Moitra, A., Rigollet, P., and Urschel, J. Rates
of estimation for determinantal point processes. In Con-
ference on Learning Theory (COLT), 2017.
Chen, D., Sain, S. L., and Guo, K. Data mining for the
online retail industry: A case study of RFM model-based
customer segmentation using data mining. Journal of
Database Marketing & Customer Strategy Management,
2012.
Chen, L., Zhang, G., and Zhou, H. Fast Greedy MAP Infer-
ence for Determinantal Point Process to Improve Recom-
mendation Diversity. In Neural Information Processing
Systems (NeurIPS), 2018.
Chen, W. and Ahmed, F.
PaDGAN: A Generative Ad-
versarial Network for Performance Augmented Diverse
Designs. Journal of Mechanical Design, 143(3):031703,
2021.
Civril, A. and Magdon-Ismail, M. On selecting a maxi-
mum volume sub-matrix of a matrix and related problems.
Theoretical Computer Science, 410(47-49):4801–4811,
2009.
Derezinski, M. and Mahoney, M. W. Determinantal point
processes in randomized numerical linear algebra. No-
tices of the American Mathematical Society, 68(1):34–45,
2021.

## Page 10

One-Pass Algorithms for MAP Inference of NDPPs
Fiedler, M. and Pták, V. Some generalizations of positive
definiteness and monotonicity. Numerische Mathematik,
9(2):163–172, 1966.
Gartrell, M., Brunel, V.-E., Dohmatob, E., and Krichene, S.
Learning Nonsymmetric Determinantal Point Processes.
In Neural Information Processing Systems (NeurIPS),
2019.
Gartrell, M., Han, I., Dohmatob, E., Gillenwater, J., and
Brunel, V.-E. Scalable Learning and MAP Inference
for Nonsymmetric Determinantal Point Processes. In
International Conference on Learning Representations
(ICLR), 2021.
Gillenwater, J., Kulesza, A., and Taskar, B. Near-Optimal
MAP Inference for Determinantal Point Processes. In
Neural Information Processing Systems (NIPS), 2012.
Gillenwater, J., Kulesza, A., Mariet, Z., and Vassilvitskii, S.
Maximizing Induced Cardinality Under a Determinantal
Point Process. In Neural Information Processing Systems
(NeurIPS), 2018.
Han, I. and Gillenwater, J. MAP Inference for Customized
Determinantal Point Processes via Maximum Inner Prod-
uct Search. In International Conference on Artificial
Intelligence and Statistics (AISTATS), 2020.
Han, I., Kambadur, P., Park, K., and Shin, J. Faster Greedy
MAP Inference for Determinantal Point Processes. In
International Conference on Machine Learning, 2017.
Hough, J. B., Krishnapur, M., Peres, Y., and Virág, B. De-
terminantal Processes and Independence. Probability
surveys, 3:206–229, 2006.
Indyk, P., Mahabadi, S., Oveis Gharan, S., and Rezaei, A.
Composable Core-sets for Determinant Maximization: A
Simple Near-Optimal Algorithm. In International Con-
ference on Machine Learning (ICML), 2019.
Indyk, P., Mahabadi, S., Oveis Gharan, S., and Rezaei, A.
Composable Core-sets for Determinant Maximization
Problems via Spectral Spanners. In Proceedings of the
Fourteenth Annual ACM-SIAM Symposium on Discrete
Algorithms (SODA), 2020.
Instacart.
The Instacart Online Grocery Shopping
Dataset, 2017.
URL https://www.instacart.
com/datasets/grocery-shopping-2017. Ac-
cessed August 2021.
Karp, R. M. On-Line Algorithms Versus Off-Line Algo-
rithms: How Much is It Worth to Know the Future? In
Proceedings of the IFIP 12th World Computer Congress
on Algorithms, Software, Architecture - Information Pro-
cessing, 1992.
Karp, R. M., Vazirani, U. V., and Vazirani, V. V. An Optimal
Algorithm for On-Line Bipartite Matching. In Proceed-
ings of the Twenty-Second Annual ACM Symposium on
Theory of Computing (STOC), 1990.
Ko, C.-W., Lee, J., and Queyranne, M. An Exact Algorithm
for Maximum Entropy Sampling. Operations Research,
1995.
Krause, A. and Golovin, D. Submodular Function Maxi-
mization. In Tractability: Practical Approaches to Hard
Problems. Cambridge University Press, February 2014.
Kulesza, A. and Taskar, B. k-DPPs: Fixed-size Determi-
nantal Point Processes. In International Conference on
Machine Learning (ICML), 2011.
Kulesza, A. and Taskar, B. Determinantal Point Processes
for Machine Learning. Foundations and Trends® in Ma-
chine Learning, 2012.
Lang, H., Sontag, D., and Vijayaraghavan, A. Optimality of
Approximate Inference Algorithms on Stable Instances.
In International Conference on Artificial Intelligence and
Statistics, pp. 1157–1166, 2018.
Lang, H., Sontag, D., and Vijayaraghavan, A. Block Sta-
bility for MAP Inference. In The 22nd International
Conference on Artificial Intelligence and Statistics, pp.
216–225, 2019.
Lang, H., Reddy, A., Sontag, D., and Vijayaraghavan, A.
Beyond Perturbation Stability: LP Recovery Guarantees
for MAP Inference on Noisy Stable Instances. In Interna-
tional Conference on Artificial Intelligence and Statistics,
pp. 3043–3051. PMLR, 2021a.
Lang, H., Sontag, D., and Vijayaraghavan, A. Graph cuts
always find a global optimum for Potts models (with a
catch). In International Conference on Machine Learning,
pp. 5990–5999. PMLR, 2021b.
Launay, C., Desolneux, A., and Galerne, B. Determinantal
point processes for image processing. SIAM Journal on
Imaging Sciences, 14(1):304–348, 2021.
Liu, P., Soni, A., Kang, E. Y., Wang, Y., and Parsana, M.
Diversity on the Go! Streaming Determinantal Point Pro-
cesses under a Maximum Induced Cardinality Objective.
In Proceedings of the Web Conference (WWW), 2021.
Macchi, O. The Coincidence Approach to Stochastic Point
Processes. Advances in Applied Probability, 7(1):83–122,
1975.
Mahabadi, S., Razenshteyn, I., Woodruff, D. P., and Zhou, S.
Non-Adaptive Adaptive Sampling on Turnstile Streams.
In Proceedings of the 52nd Annual ACM SIGACT Sympo-
sium on Theory of Computing (STOC), 2020.

## Page 11

One-Pass Algorithms for MAP Inference of NDPPs
Makarychev, K., Makarychev, Y., and Vijayaraghavan, A.
Bilu–Linial stable instances of max cut and minimum
multiway cut. In Proceedings of the twenty-fifth annual
ACM-SIAM symposium on Discrete algorithms, pp. 890–
906. SIAM, 2014.
Makarychev, K., Reddy, A., and Shan, L. Improved guaran-
tees for k-means++ and k-means++ parallel. Advances
in Neural Information Processing Systems, 33:16142–
16152, 2020.
McFee, B., Bertin-Mahieux, T., Ellis, D. P., and Lanckriet,
G. R. The Million Song Dataset Challenge. In Proceed-
ings of the International Conference on World Wide Web
(WWW), 2012.
Mirzasoleiman, B., Badanidiyuru, A., Karbasi, A., Vondrák,
J., and Krause, A. Lazier Than Lazy Greedy. In Proceed-
ings of the AAAI Conference on Artificial Intelligence
(AAAI), 2015.
Muthukrishnan, S. Data streams: Algorithms and applica-
tions. Now Publishers Inc, 2005.
Nemhauser, G., Wolsey, L., and Fisher, M. An Analysis of
Approximations for Maximizing Submodular Set Func-
tions I. Mathematical Programming, 14(1), 1978.
Nguyen, V., Le, T., Yamada, M., and Osborne, M. A. Opti-
mal Transport Kernels for Sequential and Parallel Neural
Architecture Search. In International Conference on Ma-
chine Learning (ICML), 2021.
Perez-Beltrachini, L. and Lapata, M. Multi-Document Sum-
marization with Determinantal Point Process Attention.
Journal of Artificial Intelligence Research (JAIR), 71:
371–399, 2021.
Perez-Nieves, N., Yang, Y., Slumbers, O., Mguni, D. H.,
Wen, Y., and Wang, J. Modelling Behavioural Diversity
for Learning in Open-Ended Games. In International
Conference on Machine Learning (ICML), 2021.
Qian, X., Rossi, R. A., Du, F., Kim, S., Koh, E., Malik,
S., Lee, T. Y., and Chan, J. Learning to Recommend
Visualizations from Data. In Proceedings of the 27th
ACM SIGKDD Conference on Knowledge Discovery &
Data Mining, pp. 1359–1369, 2021.
Roughgarden, T. Beyond the Worst-Case Analysis of Algo-
rithms. Cambridge University Press, 2021.
Sharma, M., Harper, F. M., and Karypis, G. Learning from
Sets of Items in Recommender Systems. ACM Transac-
tions on Interactive Intelligent Systems (TiiS), 9(4):1–26,
2019.
Shrivastava, A., Song, Z., and Xu, Z. Sublinear Least-
Squares Value Iteration via Locality Sensitive Hashing.
arXiv preprint arXiv:2105.08285, 2021.
Song, Z. and Yu, Z. Oblivious Sketching-based Central
Path Method for Solving Linear Programming Problems.
In 38th International Conference on Machine Learning
(ICML), 2021.
Song, Z., Yang, S., and Zhang, R. Does Preprocessing
Help Training Over-parameterized Neural Networks?
Advances in Neural Information Processing Systems
(NeurIPS), 34, 2021a.
Song, Z., Zhang, L., and Zhang, R. Training Multi-Layer
Over-Parametrized Neural Network in Subquadratic Time.
arXiv preprint arXiv:2112.07628, 2021b.
Xu, Z., Song, Z., and Shrivastava, A. Breaking the Lin-
ear Iteration Cost Barrier for Some Well-known Condi-
tional Gradient Methods Using MaxIP Data-structures.
Advances in Neural Information Processing Systems
(NeurIPS), 34, 2021.

## Page 12

One-Pass Algorithms for MAP Inference of NDPPs
A. Streaming MAP Inference Details
Theorem 2. For a random-order arrival stream, if S is the solution output by Algorithm 1 at the end of the stream and
σmin > 1 where σmin and σmax denote the smallest and largest singular values of LS among all S ⊆[n] and |S| ≤2k,
then
E[log det(LS)]
log(OPT)
≥
 
1 −
1
σ
(1−1
e )·(2 log σmax−log σmin)
min
!
where LS = V ⊤
S VS + B⊤
S CBS and OPT =
max
R⊆[n], |R|=k det(V ⊤
R VR + B⊤
RCBR).
Proof. As described in Algorithm 1, we will use S0, S1, . . . , Sk to denote the solution sets maintained by the algorithm,
where Si represents the solution set of size i. In particular, we have that Si = Si−1 ∪{si} where si = arg maxj∈Bi f(S ∪
{j}) and Bi denotes the i’th partition i.e. Bi := { (i−1)·n
k
+ 1, (i−1)·n
k
+ 2, . . . , i·n
k }.
For i ∈[k], let us use Xi := [Bi ∩(S∗\ Si−1) ̸= ∅] to denote the event that there is at least one element of the optimal
solution which has not already been picked by the algorithm in the batch Bi and λi := |S∗\ Si−1|. Then,
Pr[Xi] = 1 −Pr[Xc
i ]
= 1 −(1 −λi
n )(1 −
λi
n −1) . . . (1 −
λi
n −n
k + 1)
≥1 −

1 −λi
n
 n
k
≥1 −e−λi
k
≥λi
k ·

1 −1
e

Here we use the facts: ex ≥1 + x for all x ∈R, 1 −e−λ
k is concave as a function of λ, and λ ∈[0, k].
For any element s ∈[n] and set S ⊆[n], let us use f(s | S) := f(S ∪{s}) −f(S) to denote the marginal gain in f obtained
by adding the element s to the set S. For any round i ∈[k], we then have that f(Si) −f(Si−1) = f(si | Si−1).
Note that
E[f(si | Si−1) | Xi] ≥
P
ω∈OPT\Si−1 f(ω | Si−1)
|OPT \ Si−1|
.
This happens due to the fact that conditioned on Xi, every element in S∗\ Si−1 is equally likely to be present in Bi and the
algorithm picks si such that f(si | Si−1) ≥f(s | Si−1) for all s ∈Bi.
E[f(si | Si−1) | Si−1] = E[f(si | Si−1) | Si−1, Xi] Pr[Xi]
+ E[f(si | Si−1) | Si−1, Xc
i ] Pr[Xc
i ]
≥E[f(si | Si−1) | Si−1, Xi] Pr[Xi]
≥λi
k

1 −1
e

·
P
ω∈S∗\Si−1 f(ω | Si−1)
|S∗\ Si−1|
=
λi
|S∗\ Si−1|

1 −1
e

· 1
k ·
X
ω∈S∗\Si−1
f(ω | Si−1)
=

1 −1
e

· 1
k ·
X
ω∈S∗\Si−1
f(ω | Si−1)
≥

1 −1
e

· 1
k · γ · (f(Si−1 ∪S∗) −f(Si−1))
≥

1 −1
e

· 1
k · γ · (OPT −f(Si−1))

## Page 13

One-Pass Algorithms for MAP Inference of NDPPs
For the last 2 inequalities, we use the fact that f(S) = log det(LS) is monotone non-decreasing and has a submodularity
ratio of γ =

2 log σmax
log σmin −1
−1
when σmin > 1 (Gartrell et al., 2021)[Eq. 45].
Taking expectation over all random draws of Si−1, we get
E[f(si | Si−1)] ≥

1 −1
e

· γ
k (OPT −E[f(Si−1)])
Combining the above equation with f(si|Si−1) = f(Si) −f(Si−1), we have
E[f(Si)] −E[f(Si−1)] ≥

1 −1
e

· γ
k · (OPT −E[f(Si−1)])
Next we have
−(OPT −E[f(Si)]) + (OPT −E[f(Si−1)]) ≥

1 −1
e

· γ
k · (OPT −E[f(Si−1)])
Re-organizing the above equation, we obtain
OPT −E[f(Si)] ≤

1 −

1 −1
e

· γ
k

(OPT −E[f(Si−1)])
Applying the above equation recursively k times,
OPT −E[f(Sk)] ≤

1 −

1 −1
e

· γ
k
k
(OPT −E[f(S0)])
=

1 −

1 −1
e

· γ
k
k
OPT
where the last step follows from f(S0) = 0.
Re-organized the terms again, we have
E[f(Sk)] ≥
 
1 −

1 −

1 −1
e

· γ
k
k!
OPT
≥

1 −e−γ(1−1
e )
OPT
When we substitute γ =

2 log σmax
log σmin −1
−1
, we get our final inequality:
E[f(Sk)] ≥
 
1 −
1
σ
(1−1
e )·(2 log σmax−log σmin)
min
!
OPT
■

## Page 14

One-Pass Algorithms for MAP Inference of NDPPs
B. Hard instance for One-Pass MAP Inference of NDPPs
Outline: We will now give a high-level description of a hard instance for online MAP inference of NDPPs (this is inspired
by (Anari & Vuong, 2022, Example 5)) . Suppose we have a total of 2d items consisting of pairs of complementary items
like modem-router, printer-ink cartridge, pencil-eraser etc. Let us use {1, 1c, 2, 2c, . . . , d, dc} to denote them. Any item i
is independent of every item other than it’s complement ic. Individually, Pr[{i}] = Pr[{ic}] = 0 . And Pr[{i, ic}] = x2
i
with xi > 0 for all i ∈[d]. Also, we have Pr[{i, j}] = 0 for any i ̸= j. Suppose any of our online algorithms are given the
sequence {1, 2, 3, . . . , d, rc} where r ∈[d] is some arbitrary item and the algorithm needs to pick 2 items i.e. k = 2. Then,
OPT > 0 whereas all of our online algorithms (Online LSS, Online 2-neighbor, Online-Greedy) will fail to output a valid
solution.
Details: Let 0 < x1 < x2 < · · · < xd. Suppose
C =


0
x1
−x1
0
0
x2
−x2
0
...
0
xd
−xd
0


C ∈R2d×2d is a skew-symmetric (i.e. C = −C⊤) block diagonal matrix where the blocks are of the form
 0
xi
−xi
0

.
Suppose we have a total of 2d items consisting of d pairs of complementary items. We use {1, 1c, 2, 2c, . . . , d, dc} to denote
them. Let vi = vic = 0 ∀i ∈[d] and b1 = e1, b1c = e2, . . . , bdc = e2d where e1, e2, . . . , e2d are the standard unit vectors
in R2d i.e. B = I2d.
For a pair of complementary items S = {i, ic}, f(S) = x2
i . Without loss of generality, consider S = {1, 1c}. Then we can
compute B⊤
S CBS as follows:
B⊤
S CBS =

e1
e2
⊤C

e1
e2

=

0
x1
0
0
· · ·
0
−x1
0
0
0
· · ·
0

·
e1
e2

=
 0
x1
−x1
0

In this case, we have f(S) = x2
1.
For any pair of non-complementary items S = {i1, i2} where the indices are distinct, f(S) = 0. Without loss of generality,
we can consider S = {1, 2}. Then,
B⊤
S CBS =
e1
e3
⊤C
e1
e3

=
0
x1
0
0
· · ·
0
0
0
0
x2
· · ·
0

·
e1
e3

=
0
0
0
0

And so, we have that f(S) = 0.

## Page 15

One-Pass Algorithms for MAP Inference of NDPPs
C. Experiments and Datasets details
All experiments were performed using a standard desktop computer (Quad-Core Intel Core i7, 16 GB RAM) using many
real-world datasets composed of sets (or baskets) of items from some ground set of items:
• UK Retail: This is an online retail dataset consisting of sets of items all purchased together by users (in a single
transaction) (Chen et al., 2012). There are 19,762 transactions (sets of items purchased together) that consist of 3,941
items. Transactions with more than 100 items are discarded.
• MovieLens: This dataset contains sets of movies that users watched (Sharma et al., 2019). There are 29,516 sets
consisting of 12,549 movies.
• Amazon Apparel: This dataset consists of 14,970 registries (sets) from the apparel category of the Amazon Baby
Registries dataset, which is a public dataset that has been used in prior work on NDPPs (Gartrell et al., 2021; 2019).
These apparel registries are drawn from 100 items in the apparel category.
• Amazon 3-category: We also use a dataset composed of the apparel, diaper, and feeding categories from Amazon
Baby Registries, which are the most popular categories, giving us 31,218 registries made up of 300 items (Gartrell
et al., 2019).
• Instacart: This dataset represents sets of items purchased by users on Instacart (Instacart, 2017). Sets with more than
100 items are ignored. This gives 3.2 million total item-sets from 49,677 unique items.
• Million Song: This is a dataset of song playlists put together by users where every playlist is a set (basket) of songs
played by Echo Nest users (McFee et al., 2012). Playlists with more than 150 songs are discarded. This gives 968,674
playlists from 371,410 songs.
• Customer Dashboards: This dataset consists of dashboards or baskets of visualizations created by users (Qian et al.,
2021). Each dashboard represents a set of visualizations selected by a user. There are 63436 dashboards (baskets/sets)
consisting of 1206 visualizations.
• Web Logs: This dataset consists of sets of webpages (baskets) that were all visited in the same session. There are 2.8
million baskets (sets of webpages) drawn from 2 million webpages.
• Company Retail: This dataset contains the set of items viewed (or purchased) by a user in a given session. Sets
(baskets) with more than 100 items are discarded. This results in 2.5 million baskets consisting of 107,349 items.
The last two datasets are proprietary Adobe data. The learning algorithm of (Gartrell et al., 2021) takes as input a parameter
d, which is the embedding size for V , B, C. We use d = 10 for all datasets other than Instacart, Customer Dashboards,
Company Retail where d = 50 is used and Million Song, where d = 100 is used. For all of our results in 7, we set k = 8
and choose α = 1.1.
D. Additional Experimental Results
D.1. Number of Determinant Computations
We perform several experiments comparing the number of determinant computations (as a system-independent proxy for
time) of all our online algorithms. We do not compare with offline greedy here because that algorithm doesn’t explicitly
compute all the determinants. Results comparing the number of determinant computations as a function of the number of
data points analyzed for a variety of datasets are provided in Figure 2. Online-2-neighbor requires the most number of
determinant computations but also gives the best results in terms of solution value. Online-LSS and Online-Greedy use very
similar number of determinant computations.

## Page 16

One-Pass Algorithms for MAP Inference of NDPPs
0
500
1000 1500 2000 2500 3000 3500 4000
Data points analyzed
0
25000
50000
75000
100000
125000
150000
175000
200000
Det. computations
UK Retail
Online 2-neighbor
Online LSS
Online Greedy
0
2000
4000
6000
8000
10000 12000
Data points analyzed
0
100000
200000
300000
400000
500000
600000
Det. computations
MovieLens
Online 2-neighbor
Online LSS
Online Greedy
20
40
60
80
100
Data points analyzed
0
2000
4000
6000
8000
10000
12000
14000
Det. computations
Amazon Apparel
Online 2-neighbor
Online LSS
Online Greedy
0
50
100
150
200
250
300
Data points analyzed
0
10000
20000
30000
40000
50000
Det. computations
Amazon 3-category
Online 2-neighbor
Online LSS
Online Greedy
0
10000
20000
30000
40000
50000
Data points analyzed
0
500000
1000000
1500000
2000000
2500000
3000000
3500000
Det. computations
instacart
Online 2-neighbor
Online LSS
Online Greedy
0
50000 100000150000200000250000300000350000
Data points analyzed
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
Det. computations
1e7
Million Song
Online 2-neighbor
Online LSS
Online Greedy
0
200
400
600
800
1000
1200
Data points analyzed
0
10000
20000
30000
40000
Det. computations
Customer Dashboards
Online 2-neighbor
Online LSS
Online Greedy
Company Retail
Figure 2. Results comparing the number of determinant computations as a function of the number of data points analyzed for all our
online algorithms. Online-2-neighbor requires the most number of determinant computations but also gives the best results in terms of
solution value. Online-LSS and Online-Greedy use very similar number of determinant computations.

## Page 17

One-Pass Algorithms for MAP Inference of NDPPs
D.2. Number of Swaps
Results comparing the number of swaps (as a measure of solution consistency) of all our online algorithms can be found in
Figure 3. Online-Greedy has the most number of swaps and therefore the least consistent solution set. On most datasets, the
number of swaps by Online-2-neighbor is very similar to Online-LSS.
0
500
1000 1500 2000 2500 3000 3500 4000
Data points analyzed
0
10
20
30
40
Swaps
UK Retail
Online 2-neighbor
Online LSS
Online Greedy
0
2000
4000
6000
8000
10000
12000
Data points analyzed
0
10
20
30
40
50
Swaps
MovieLens
Online 2-neighbor
Online LSS
Online Greedy
20
40
60
80
100
Data points analyzed
0.0
2.5
5.0
7.5
10.0
12.5
15.0
17.5
Swaps
Amazon Apparel
Online 2-neighbor
Online LSS
Online Greedy
0
50
100
150
200
250
300
Data points analyzed
0
5
10
15
20
25
30
Swaps
Amazon 3-category
Online 2-neighbor
Online LSS
Online Greedy
0
10000
20000
30000
40000
50000
Data points analyzed
0
20
40
60
80
100
120
Swaps
instacart
Online 2-neighbor
Online LSS
Online Greedy
0
50000 100000150000200000250000300000350000
Data points analyzed
0
20
40
60
80
100
Swaps
Million Song
Online 2-neighbor
Online LSS
Online Greedy
0
200
400
600
800
1000
1200
Data points analyzed
0
2
4
6
8
10
Swaps
Customer Dashboards
Online 2-neighbor
Online LSS
Online Greedy
Company Retail
Figure 3. Results comparing the number of swaps of all our online algorithms. Online-Greedy does the most number of swaps and
therefore has the least consistent solution set. On most datasets, the number of swaps by Online-2-neighbor is very similar to Online-LSS.
D.3. Random Streams
We also investigate our algorithms under the random stream paradigm. For this setting, we use some of the previous
real-world datasets, and randomly permute the order in which the data appears in the stream. We do this 100 times and
report the average of solution values in Figure 4 and the average of number of determinant computations and swaps in
Figure 5. We observe that Online-2-neighbor and Online-LSS give very similar performance in this regime and they are
always better than Online-Greedy.
D.4. Ablation study varying ϵ
To study the effect of ϵ in Online-LSS (Algorithm 2), we vary ϵ ∈{0.05, 0.1, 0.3, 0.5} and analyze the value of the obtained
solutions, number of determinant computations, and number of swaps. We notice that, in general, the solution quality,
number of determinant computations, and the number of swaps increase as ϵ decreases. Results are provided in Figure 6.

## Page 18

One-Pass Algorithms for MAP Inference of NDPPs
0
50
100
150
200
250
300
Data points analyzed
0.0e+00
2.0e-11
4.0e-11
6.0e-11
8.0e-11
1.0e-10
Solution
Amazon 3-category
Online 2-neighbor
Online LSS
Online Greedy
Offline
20
40
60
80
100
Data points analyzed
0.0e+00
2.5e-10
5.0e-10
7.5e-10
1.0e-09
1.3e-09
1.5e-09
1.8e-09
2.0e-09
Solution
Amazon Apparel
Online 2-neighbor
Online LSS
Online Greedy
Offline
Figure 4. Solution quality as a function of the number of data points analyzed in the random stream paradigm. Online-2-neighbor and
Online-LSS give very similar performance in this setting and they are always better than Online-Greedy.
0
50
100
150
200
250
300
Data points analyzed
0
10000
20000
30000
40000
50000
60000
Det. computations
Amazon 3-category
Online 2-neighbor
Online LSS
Online Greedy
0
50
100
150
200
250
300
Data points analyzed
0
5
10
15
20
25
30
35
Swaps
Amazon 3-category
Online 2-neighbor
Online LSS
Online Greedy
Figure 5. Number of determinant computations and swaps as a function of the number of data points analyzed in the random stream
setting. Online-2-Neighbor needs more determinant computations than Online-LSS but has very similar number of swaps in this setting.
Note that ϵ = α −1
.

## Page 19

One-Pass Algorithms for MAP Inference of NDPPs
0
500
1000 1500 2000 2500 3000 3500 4000
Data points analyzed
0.0
0.5
1.0
1.5
2.0
2.5
Solution
1e−8
UK Retail
ε = 0.05
ε = 0.1
ε = 0.3
ε = 0.5
0
500
1000 1500 2000 2500 3000 3500 4000
Data points analyzed
0
5000
10000
15000
20000
25000
30000
35000
40000
Det. computations
UK Retail
ε = 0.05
ε = 0.1
ε = 0.3
ε = 0.5
0
500
1000 1500 2000 2500 3000 3500 4000
Data points analyzed
0
5
10
15
20
25
Swaps
UK Retail
ε = 0.05
ε = 0.1
ε = 0.3
ε = 0.5
20
40
60
80
100
Data points analyzed
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
Solution
1e−9
Amazon Apparel Reg.
ε = 0.05
  = 0.1
  = 0.3
  = 0.5
20
40
60
80
100
Data points analyzed
0
200
400
600
800
1000
1200
1400
Det. computations
Amazon Apparel Reg.
ε = 0.05
ε = 0.1
ε = 0.3
ε = 0.5
20
40
60
80
100
Data points analyzed
0
2
4
6
8
10
Swaps
Amazon Apparel Reg.
ε = 0.05
ε = 0.1
ε = 0.3
ε = 0.5
0
50
100
150
200
250
300
Data points analyzed
0
1
2
3
4
Solution
1e−11
Amazon Baby Reg.
ε = 0.05
ε = 0.1
ε = 0.3
ε = 0.5
0
50
100
150
200
250
300
Data points analyzed
0
1000
2000
3000
4000
5000
Det. computations
Amazon Baby Reg.
ε = 0.05
ε = 0.1
ε = 0.3
ε = 0.5
0
50
100
150
200
250
300
Data points analyzed
0
5
10
15
20
Swaps
Amazon Baby Reg.
ε = 0.05
ε = 0.1
ε = 0.3
ε = 0.5
Figure 6. Performance of Online-LSS varying ϵ for k = 8. Solution quality, number of determinant computations, and number of swaps
seem to increase with decreasing ϵ.

## Page 20

One-Pass Algorithms for MAP Inference of NDPPs
D.5. Ablation study varying set size k and ϵ
In this set of experiments on the Amazon-Apparel dataset using Online-LSS, we study the choice of set size k and ϵ on the
solution quality, number of determinant computations, and number of swaps while fixing all other settings to be same as
those used previously in Figure 4. We can see that as k increases, the solution value decreases across all values of ϵ. This is
in accordance with our intuition that the probability of larger sets should be smaller. In general, the number of determinant
computations and swaps increases for increasing k across different values of ϵ.
0.05
0.1
0.2
0.3
0.4
0.5
5
6
7
8
9
10
(a) Solution
0.05
0.1
0.2
0.3
0.4
0.5
(b) # det computations
0.05
0.1
0.2
0.3
0.4
0.5
(c) # swaps
Figure 7. Comparing the effect of set size k and ϵ on the solution, number of determinant computations, and number of swaps for
ONLINE-LSS.
