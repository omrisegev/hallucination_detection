---
source_pdf: C:/Users/omris/TAU/hallucination_detection/papers/Task-Based Graph Signal Compression.pdf
slug: task-based-graph-signal-compression
pages: 13
extracted_on: 2026-09-07
---

# Task-Based Graph Signal Compression

## Page 1

Task-Based Graph Signal Compression
Pei Li, Nir Shlezinger, Haiyang Zhang, Baoyun Wang, and Yonina C. Eldar
Abstract—Graph signals arise in various applications, ranging
from sensor networks to social media data. The high-dimensional
nature of these signals implies that they often need to be
compressed in order to be stored and transmitted. The common
framework for graph signal compression is based on sampling,
resulting in a set of continuous-amplitude samples, which in
turn have to be quantized into a ﬁnite bit representation. In
this work we study the joint design of graph signal sampling
along with quantization, for graph signal compression. We focus
on bandlimited graph signals, and show that the compression
problem can be represented as a task-based quantization setup,
in which the task is to recover the spectrum of the signal. Based
on this equivalence, we propose a joint design of the sampling
and recovery mechanisms for a ﬁxed quantization mapping,
and present an iterative algorithm for dividing the available bit
budget among the discretized samples. Furthermore, we show
how the proposed approach can be realized using graph ﬁlters
combining elements corresponding the neighbouring nodes of the
graph, thus facilitating distributed implementation at reduced
complexity. Our numerical evaluations on both synthetic and
real world data shows that the joint sampling and quantization
method yields a compact ﬁnite bit representation of high-
dimensional graph signals, which allows reconstruction of the
original signal with accuracy within a small gap of that achievable
with inﬁnite resolution quantizers.
I. INTRODUCTION
Rapid development of information and communication tech-
nology, has lead to a growing need to process and store high-
dimensional signals [2]. In many families of signals, such
as those representing communication networks, social media
data, and sensors deployment, the interactions between the
elements of the signal obey some graphical structure. Graph
signal processing (GSP) has emerged as a promising technique
to deal with such complex signals [3], [4].
The high-dimensional nature of graph signals gives rise
to the need to compressing them [5]. A leading strategy to
compress graph signals is based on sampling their elements
[6]–[8]. Sampling techniques typically build upon the fre-
quency analysis of graph signals, which is a fundamental
tool in GSP, utilized also for processing such as denoising
[9], [10] and interpolation [11]. Generally speaking, the graph
signal values associated with the two end vertices of edges
with large weights in the graph tend to be similar [12]. This
property, often observed in practice, leads to spectral sparsity,
and the resulting signals are referred to as bandlimited [13].
Part of this work has been presented in the IEEE International Conference
on Acoustics, Speech, and Signal Processing (ICASSP), 2021 [1]. This work
was supported in part by the National Natural Science Foundation of China
under grant No 61971238, by the European Research Council (ERC) under
the European Union’s Horizon 2020 research and innovation programme, and
by the Israel Science Foundation under grant No. 0100101. P. Li and B.
Wang are with the School of Communication and Information Engineering,
Nanjing University of Posts and Telecommunications, Nanjing, China (e-
mail: {2017010211; bywang}@njupt.edu.cn) N. Shlezinger is with the School
of ECE, Ben-Gurion University of the Negev, Beer-Sheva, Israel (e-mail:
nirshl@bgu.ac.il) H. Zhang and Y. C. Eldar are with the Faculty of Math and
CS, Weizmann Institute of Science, Rehovot, Israel (e-mail: {haiyang.zhang;
yonina.eldar}@weizmann.ac.il).
Basic graph sampling theory focuses on bandlimited graph
signals and relates the spectral support to the number of
samples required for representing the signal such that it can
be reconstructed from noiseless samples [6]. Nonetheless, in
the presence of noise, it is often challenging to determine
how to select which nodes to sample. In general, sampling
set selection is an NP-hard issue, which is often tackled using
greedy approaches [14]. Recently, sampling theorems for non-
bandlimited graph signals, exploiting sparsity in domains other
than the spectral domain, were proposed in [7], [8].
A key characteristic of graph signal compression by sam-
pling stems from the fact that it represents the signal by a set of
continuous-amplitude samples, where the number of samples
is treated as the compression dimension. However, when stor-
ing and transmitting graph signals, the level of compression
is measured in the number of bits, rather than continuous-
amplitude samples, required for representing the signal. This
implies that graph signal compression should involve not only
sampling, but also quantization [15]. To date, existing works
on quantization in GSP, e.g., [16]–[20], focus on applying GSP
methods to graph signals with quantized elements [16]–[19] or
quantizing sampled graph signals [20], rather than designing
joint sampling and quantization methods for compressing such
signals. This motivates the design of graph signal compression
schemes which combine both sampling as well as quantization
designed in a joint manner as papers on ADC compression.
In this work, we propose a compression method for ban-
dlimited graph signals which maps a high-dimensional signal
into a ﬁnite-bit representation via sampling and quantiza-
tion. Our compression method is inspired by the recently
proposed task-based quantization framework [21]–[26]. Task-
based quantization studies the acquisition of multivariate
continuous-amplitude signals in order to recover some un-
derlying information vector, while operating under an overall
bit budget. This is achieved by processing the signal using
a dimensionality-reducing combining mapping, carried out in
the analog domain, followed by scalar quantization and digital
processing, all jointly designed to recover the task vector [25].
Our ability to treat graph signal compression as task-based
quantization stems from the observation that graph signal
sampling can be viewed as an analog combining operation,
while the task in reconstructing bandlimited graph signals
is the recovery of their spectrum. Unlike the original task-
based quantization formulation, which focused on the design
of analog-to-digital convertors (ADCs), and thus considered
identical quantization mappings applied to each sampled el-
ement, here we consider a compression problem rather than
signal acquisition, and thus does not enforce this restriction.
Our joint sampling and quantization design employs a graph
ﬁlter prior to the quantization operation. We consider two
types of such graph ﬁlters. The ﬁrst is an unconstrained
graph ﬁlter, which is allowed to carry out arbitrary linear
operations on the graph signals during the sampling procedure.
1
arXiv:2110.12387v1  [eess.SP]  24 Oct 2021

## Page 2

For the unconstrained setup, we begin by studying the case in
which the number of bits used for representing each sample
is ﬁxed, and derive the sampling operator that minimizes the
recovery mean-squared error (MSE). Then, we consider an
overall bit budget, and optimize the number of bits assigned
to each sample. We identify a sufﬁcient condition for which
the allocation is optimized by using identical quantizers, i.e.,
as in conventional task-based quantization [21].
The second considered ﬁlter corresponds to frequency do-
main graph ﬁlters [8]. These ﬁlters are constrained to represent
local computations, such that each node can be combined
only with neighbouring nodes. Frequency domain graph ﬁl-
ters notably facilitate distributed implementation, compared to
unconstrained graph samplers which may require each sample
to be produced by observing the entire graph [27]. For this
constrained family, where the sampling operation is modeled
by the selection of a subset of the outputs of a frequency
domain graph ﬁlter, we ﬁrst derive the sampling set and the
corresponding assigned number of bits. Then we propose an
alternating optimization algorithm to design the ﬁlter along
with the overall compression system.
Our simulation study considers graph signal compression
in three different applications: signals representing sensor net-
works; meteorological real-world data; and natural images. We
consistently demonstrate that the proposed algorithm results
in a distortion which is within a minor gap of that achievable
without bit constraints, while yielding notably more accurate
representations of the graph signal under a given bit budget-
compared to separately designed sampling and quantization.
The rest of the paper is organized as follows: Section II
introduces the graph signal model, and formulates the com-
pression problem. In Section III, the compression scheme is
presented for generic graph ﬁlters, while frequency domain
graph ﬁlters are considered in Section IV. Section V details
numerical simulations. Finally, Section VI provides concluding
remarks. Detailed proofs are delegated to the Appendix.
Throughout the paper, we use lower-case (upper-case) bold
characters to denote vectors (matrices). The transpose, pseudo-
inverse and trace of a matrix A are respectively denoted as
AT , A† and Tr(A), while (A)i,j is the (i, j)th entry of A,
and AS represents a sub-matrix of A with rows indexed by S.
For a vector a, ai is its i-th element, and diag(a) is a diagonal
matrix with a on its main diagonal. For a scalar a, ⌈a⌉and
⌊a⌋represent the round up and round down for a, respectively.
We use R for the set of real numbers, Z for the integers, and
1A is the indicator function for event A.
II. SYSTEM MODEL
In this section we present the system model for which
we derive the joint sampling and quantization compression
method. We ﬁrst discuss the graph signal model in Subsec-
tion II-A, followed by a brief review of graph sampling and
quantization in Subsection II-B, and a formulation of the
considered problem in Subsection II-C.
A. Bandlimited Graph Signal Model
We consider an undirected and weighted graph G
=
(V, E, W), in which V
= {v1, v2, ..., vN} is the set of
nodes and E is the set of edges. Let W ∈CN×N be the
adjacency matrix, such that its (i, j)th element Wi,j models
the similarity/relationship between nodes i and j. A graph
signal is a function f : V →RN deﬁned on the vertices
of G. We adopt the symmetric normalized Laplacian matrix
L = I −D−1/2WD−1/2 as the variation operator, where
D = diag{d1, d2, ..., dN} is the degree matrix whose ith
diagonal element is given by di = P
j Wi,j. Since L is
diagonalizable, there exists an orthogonal matrix U and a
diagonal matrix Λ satisfying L = UΛUT . For a graph signal
x ∈RN deﬁned over G, its graph Fourier transform (GFT)
is deﬁned as ˆx = UT x. The graph signal is bandlimited if
there exists an integer K ≤N such that its GFT satisﬁes
ˆxk = 0 for all k ≥K. The smallest value of K denotes the
bandwidth of f. Bandlimited graph signals can be expressed as
UKc, where UK is the ﬁrst K columns of U, while c ∈RK
is the frequency representation. Here, we consider a noisy
observation of a bandlimited graph signal, given by
x = UKc + w,
(1)
where w is an i.i.d. zero-mean noise with variance σ2
0.
As in the classical factor analysis model [28], we impose
a Gaussian prior on the spectral representation c. Speciﬁcally,
we assume that c follows a degenerate zero-mean multivariate
Gaussian distribution such that c ∼N(0, ˆΛ), where ˆΛ =
diag{σ2
1, σ2
2, ..., σ2
K}, and σ2
i represents the variance for i-
th spectral representation ci for i ∈{1, · · · , K}. Under this
model, the signal x obeys a zero-mean multivariate Gaussian
distribution with covariance matrix:
Cx = UK ˆΛUT
K + σ2
0I = U ˜ΛUT ,
(2)
where
˜Λ = diag

σ2
1 + σ2
0, σ2
2 + σ2
0, ..., σ2
K + σ2
0, σ2
0, ..., σ2
0
	
.
(3)
The joint Gaussianity of c and x implies that the minimum
MSE (MMSE) estimate of the spectrum c from the noisy graph
signal x is given by ˜c = Γ∗x, where
Γ∗= ˆΛ( ˜Λ−1)KUT .
(4)
Here, ( ˜Λ−1)K represents the ﬁrst K rows of ˜Λ−1. The
resulting estimation error is given by
E{∥˜c −c∥2} = Tr

ˆΛ −UK ˜Λ2UT
KC−1
x

.
(5)
B. Sampling and Qauntization of Graph Signals
In this work we combine sampling and quantization for
compressing bandlimited graph signals. We thus next brieﬂy
recall some basics in graph signal sampling and quantization,
starting with the deﬁnition of vertex-domain sampling:
Deﬁnition 1 (Vertex-domain sampling [8]). A vector y ∈RP
is the vertex-domain sampling of a graph signal x ∈RN, if
there exists Ψ ∈RP ×N such that y = Ψx and P < N.
Vertex-domain sampling, abbreviated henceforth as sam-
pling, often restricts Ψ to be a submatrix of the N × N
identity matrix [6]. This results in the elements of y being
also elements of x. Nonetheless, graph sampling operations
2

## Page 3

and their corresponding methods for reconstructing x from
y are studied in the literature for various forms of Ψ [8].
Sampling matrices can be generally written as Ψ = ISF,
where IS ∈RP ×N is a row-selection matrix, F ∈F ⊆RN×N
is a graph ﬁlter, and F is the set of feasible graph ﬁlters.
We separately consider two forms of graph ﬁlters. The ﬁrst
type is referred to as unconstrained graph ﬁlters:
Deﬁnition 2 (Unconstrained graph ﬁlter). An unconstrained
graph ﬁlter can be any N × N matrix, i.e., F = RN×N.
For unconstrained graph ﬁlters, the sampling matrix Ψ
can be any matrix in RP ×N. The application of sampling
using unconstrained graph ﬁlters requires the entire graph
signal to be available for generating each sample, i.e., each
element in the sampled y can be affected by every node in
the graph. Such sampling procedures can be challenging to
implement, particularly when dealing with high-dimensional
graph signals. This motivates the restriction to graph sampling
procedures which involve only local operations, where each
sample is affected only by a subset of neighbouring nodes.
Such sampling mechanisms can be realized by constraining F
to represent frequency-domain ﬁltering, deﬁned next:
Deﬁnition 3 (Frequency-domain graph ﬁlter [7]). The set of
frequency-domain graph ﬁlters deﬁned over a graph G with
Laplacian L = UΛUT is given by
F = {F = UF(Λ)UT |F(Λ) is diagonal}.
(6)
Restricting the sampling matrix Ψ to frequency-domain
ﬁltering results in the sampled vector y taking the form
y = ISUF(Λ)ˆx.
(7)
The representation of the graph sampling operation via (7)
notably facilitates distributed operation compared with sam-
pling using unconstrained graph ﬁlters. This follows since (7)
can be approximated by interpolation [29], resulting in the
frequency-domain graph ﬁlter being expressed as a matrix
polynomial function with respect to L. In particular, a fre-
quency domain graph ﬁlter restricted to combining elements
corresponding the neighbouring nodes of the graph implies
that F(Λ) should be a polynomial function with respect to Λ,
i.e., F(Λ) = PK0
i=0 βiΛi, for some coefﬁcients {βi}, where
K0 is the length of the polynomial [27].
Next, we discuss the considered quantization rule. While
in general, quantization encapsulates a broad range of
continuous-to-discrete mappings [15], here we focus on con-
ventional uniform scalar quantizaiton, deﬁned as follows:
Deﬁnition 4 (Uniform quantization). A uniform quantizer with
resolution M, support γ, and step size δ = 2γ
M , is the mapping
QM (x) ≜
(
δ
  x
δ

+ 1
2

,
if |x| < γ,
sign (x)
 γ −δ
2

,
else.
The non-linear nature of quantization mappings results in a
complex model for the distortion induced in this procedure,
i.e., the term x −QM (x). In our analysis we model the
quantization operation as non-subtractive dithered quantiza-
tion, obtained by adding noise prior to uniform quantization
[30]. The main motivation for this model is that it results in
the distortion being uncorrelated with the input signal when
the quantizer is not overloaded, i.e., |x| ≤γ. This resulting
model, which notably facilitates the design and analysis of
quantization systems, is also a good approximation of the dis-
tortion model for various quantizer input distributions without
dithering as analyzed in [31] as demonstrated in [21].
C. Problem Formulation
We consider the problem of compressing a noisy graph
signal x into a digital representation comprised of log2 M
bits. This compressed version should preserve the information
required to reconstruct the bandlimited graph signal UKc. We
assume that the structure of the graph signal and its spectral
support are known, i.e., UK is given.
Since the volume of the representation is limited in its
overall number of bits, we study compression mechanisms
consisting of both sampling and quantization, as deﬁned in
Subsection II-B. In such a system, the graph signal is ﬁrst
sampled by applying the P × N matrix Ψ, with P < N, and
the entries of the sampled y = Ψx are then discretized by
the uniform quantizers {QMi(·)}P
i=1, resulting in the ﬁnite-
bit representation Q(y) = [QM1(y1), . . . , QMP (yP )]T . The
overall bit constraint implies that PP
i=1 log2 Mi ≤log2 M.
As quantization inherently results in lossy compression [32,
Ch. 23], we do not aim to perfectly recover x, and use
the MSE as our design objective. Consequently, the joint
design of the sampling and quantization mappings can be
formulated as recovering the digital representation from which
the bandlimited portion of x, i.e., UKc, can be reconstructed
most accurately. This is mathematically formulated as
min
Ψ,Q(•)E
n
∥UKc −E {UKc|Q(Ψx)}∥2o
,
s.t.
P
X
i=1
log2 Mi ≤log2 M,
Mi ∈Z+, i ∈P,
(P1)
where P ≜{1, 2, . . . , P}. In the following sections we jointly
design the resulting compression system based on (P1).
III. JOINT SAMPLING AND QAUNTIZATION METHOD WITH
UNCONSTRAINED GRAPH FILTERS
In this section we derive joint sampling and quantization
methods for graph signal compression with unconstrained
graph ﬁlters. To that aim, we ﬁrst present in Subsection III-A
an alternative problem formulation, obtained by introducing
some relaxations to (P1). Then, we present the MSE minimiz-
ing system design in Subsection III-B, and provide a greedy-
based algorithm to tackle it in Subsection III-C.
A. Alternative Optimization Problem
The problem formulation (P1) characterizes the graph signal
compression setup as designing the sampling matrix Ψ and
the scalar quantizers Q(•). The resulting setup bears much
similarity to task-based quantization, proposed as a framework
for designing ADCs to extract information from an acquired
analog signal [21]–[24]. To exploit task-based quantization in
graph signal compression, we formulate an alternative problem
based on (P1), which can be tackled using these techniques.
3

## Page 4

First, we note that since the unitary matrix U and its
submatrix UK are known, recovering the MMSE estimate of
x, as formulated in (P1), is equivalent to the corresponding es-
timation of c. We thus henceforth refer to c as the task vector.
Furthermore, recalling the deﬁnition of ˜c = E{c|x} = Γ∗x,
then by the orthogonality principle, designing Q(•) and Ψ to
minimize the MSE w.r.t. c is the same as designing them to
minimize the MSE w.r.t. ˜c. The objective in (P1) thus becomes
min
Ψ,Q(•) E
n
∥˜c −E {˜c|Q(Ψx)}∥2o
.
(8)
Next, we relax (8) by focusing on linear recovery. Our
motivation for considering linear schemes stems from their
analytical tractability, and since they are commonly used for
reconstructing sampled graph signals [8]. The compression
system is designed such that the desired vector c can be
recovered from Q(Ψx) using a ﬁlter Φ ∈RK×P , i.e., the
recovered c is given by ˆc = ΦQ(y), where y = Ψx. This
compression and recovery system is illustrated in Fig. 1.
Quantizers are typically designed to operate within their
dynamic range [15]. Therefore, following [21], we guarantee
that the probability of overloading the quantizers is sufﬁciently
small, i.e., that Pr(|(Ψx)i| > γi) ≈0, by ﬁxing the support
of the ith quantizer to γ2
i = η2E

(Ψx)2
i
	
for each i ∈P.
For instance, setting η = 2 results in overloading probability
< 5%. When the overloading probability vanishes, then the
output of the dithered quantizers can be written as [30]
Q (Ψx) = Ψx + eQ,
(9)
where the quantization error vector eQ has i.i.d. zero-mean
entries of variance δ2
i /6 (with δi = 2γi/Mi), i.e.,
G ≜E

eQeT
Q
	
= diag
 2γ2
1
3M 2
1
, 2γ2
2
3M 2
2
, · · · , 2γ2
P
3M 2
P

.
(10)
While our setting of γ2
i achieves a small yet non-zero over-
loading probability, we design the compression mechanism by
utilizing the model in (9), which rigorously holds when the
overloading probability is zero, as proposed in [21], [22].
To summarize, the alternative optimization problem based
on which we design our scheme is given by
min
Ψ,Φ,{Mi}E
n
∥Γ∗x −Φ(Ψx + eQ)∥2o
,
s.t.
P
X
i=1
log2 Mi ≤log2 M,
Mi ∈Z+, i ∈P.
(P2)
The optimization problem (P2) can be treated as a task-based
quantization problem. In task-based quantization, an analog
ﬁlter is designed along with the overall acquisition system in
light of the system task [21]. Here, the sampling operation
implements the pre-quantization combining, and the system
task is to recover the vector c (via its MMSE estimate).
However, as task-based quantization focuses on the design
of ADCs operating in a serial manner, it is restricted to
utilizing identical quantizers, i.e., each quantizer uses the same
number of bits, while our graph signal compression problem
does not share this restriction. Therefore, our design based on
(P2), given in the following subsection, combines task-based
quantization methods with dedicated bit allocation techniques.
B. Compression System Design
In this section, we design a joint graph signal sampling
and quantization scheme based on the identiﬁed relationship
between such setups and task-based quantization in (P2).
Directly solving (P2) is difﬁcult due to the coupling between
its optimization variables, combined with the non-linear re-
lationship between these parameters and the statistical model
of the quantization error, observed in (10). Therefore, in the
following, we ﬁrst design the sampling matrix based on (P2)
for a ﬁxed bit allocation, i.e., when {Mi} is given.
As a preliminary step, we note that for any given sampling
matrix Ψ and bit allocation {Mi}, the MSE minimizing linear
recovery matrix Φ is given in the following lemma:
Lemma 1. For any ﬁxed Ψ and {Mi}, the objective in (P2)
is minimized by setting
Φ∗= Γ∗CxΨT  ΨCxΨT + G
−1 .
(11)
Proof: The lemma is obtained following [21, App. B].
Next, we design the sampling matrix Ψ. Here, we recall the
similarity between our sampling matrix and analog combiner
design in task-based quantization setups. In particular, the
MSE minimizing analog combiner for task-based quantiza-
tion is given in [21, Thm. 1] for an identical bit allocation
˜
M = ⌊M 1/P ⌋. For such setups, the MSE is minimized for
a setting of the form Ψ = UΞVT , where V ∈RN×N is
the right singular vectors matrix of Γ∗C1/2
x , Ξ ∈RP ×N is
diagonal with non-negative diagonal entries, and U ∈RP ×P
is a unitary matrix designed to balance the inputs to the
identical quantizers. Since in our setup, the quantizers are not
restricted to be identical, we cannot adopt the results derived
in [21]. Nonetheless, for a given bit allocation {Mi} sorted in
a descending order, we can characterize the MSE minimizing
sampling matrix, as stated in the following proposition:
Proposition 1. For a bit allocation {Mi} with Mi ≥Mi+1,
the MSE minimizing unconstrained sampling matrix satisﬁes
Ψ∗= UΨΞΨ ˜Λ
−1/2UT ,
(12)
where UΨ is a unitary matrix satisfying
(UΨΞΨΞT
ΨUT
Ψ)i,i = 3M 2
i
2η2 ,
(13)
and ΞΨ ∈RP ×N is a diagonal matrix with non-negative
entries. Denoting αi = (ΞΨ)2
i,i, i ∈P, the elements of ΞΨ
are obtained as the solution to the following problem:
{αi}P
i=1 = arg min
{α′
i}
P
X
i=1
λ2
Γ,i
α′
i + 1,
s.t.
3
2η2 [M 2
1 , . . . M 2
P ]T ≺[a′
1, . . . a′
P ]T ,
(14)
where ≺denotes majorization, i.e., a ≺b implies that a is
majorized by b [33]. In (14), λΓ,i is the ith largest singular
value of Γ∗C1/2
x . The resulting excess MSE is given by:
E{∥ˆc −˜c∥2} =
P
X
i=1
αiλ2
Γ,i
αi + 1.
(15)
Proof: The proof is given in Appendix A.
4

## Page 5

Fig. 1. Graph signal compression by sampling and quantization.
For the special case of identical bit allocation, i.e., Mi = Mj
for each i, j ∈P, it can be shown that Proposition 1 coincides
with [21, Thm. 1]. We note that (14) is a convex problem,
which can be solved using existing convex optimization tool-
boxes such as CVX. The majorization constraint can not be
addressed with CVX directly, So we split it into equivalent
multiple linear constraints, which are all convex and can be
solved using CVX. Consequently, while Proposition 1 does
not give the sampling matrix Ψ∗in closed form, it can
be numerically computed with an affordable computational
effort. Nonetheless, identifying the bit allocation {Mi} which
minimizes the MSE based on Proposition 1 is a challenging
task due to the discrete nature of the bit assignment, motivating
the greedy optimization algorithm detailed in the following
section. However, if one is allowed to assign non-integer bit
values, the resulting compression system and its corresponding
MSE can be obtained explicitly via the following theorem:
Theorem 1. When {Mi} are not limited to be integer, the
optimal unconstrained sampling operator is Ψ∗= UT
K. The
MSE minimizing bit allocations satisﬁes
M 2
i =
(
η2 −2β∗+λ2
Γ,i+√
λ4
Γ,i−4β∗λ2
Γ,i
3β∗
β∗≤
λ2
Γ,i
4 ,
1
otherwise,
(16)
where the hyperparameter β∗is set such that PP
i=1 log2 Mi =
log2 M. The resulting excess MSE is given by:
E{∥ˆc −˜c∥2} =
K
X
i=1
λ2
˜Γ,i −
min(K,P )
X
i=1
3M 2
i λ2
˜Γ,i
3M 2
i + 2η2 .
(17)
Proof: The proof is given in Appendix B.
While Theorem 1 considers a hypothetical system which
can quantize using non-integer number of bits, it reveals an
important challenge arising from the incorporation of quanti-
zation compared to conventional graph sampling. When one
can quantize with arbitrary non-integer levels, the resulting
compression problem reduces to a conventional graph sam-
pling (without quantization) setup, as shown in Appendix B.
For such cases, the sampling matrix Ψ∗specializes into con-
ventional frequency domain sampling of bandlimited signals,
which settles with the corresponding results in [8]. However,
since quantization is inherently restricted to discrete levels, the
MSE minimizing graph sampling matrix does not necessarily
reduce to frequency domain sampling. In such cases, one has
to utilize Proposition 1 combined with dedicated schemes for
just rounded or optimizing the bit allocation, as proposed next.
C. Greedy Optimization Algorithm Design
The MSE characterized Theorem 1 is generally not achiev-
able for graph signal compression schemes since it ignores
the fact that the bit assignment must take integer values.
Nonetheless, the closed-form MSE expression (17) can be used
to facilitate the setting of the (discrete) bit allocation {Mi}
using greedy optimization. In the following we ﬁrst introduce
the greedy algorithm for setting the bit budget, after which
we identify a sufﬁcient and necessary conditions for which
the bit assignment is designed by using identical quantizers,
and identify the number of quantizers P which minimizes the
MSE without setting any quantizer to be inactive.
1) Greedy Bit Assignment: The proposed greedy algorithm
starts by setting the minimal bit allocation for all the quantiz-
ers, i.e., M (0)
i
= 1 for each i ∈P. Then, it increments the
number of levels for the quantizer which contributes most to
the objective (17). While this procedure does not impose the
constraint Mi ≥Mi+1 for each i ∈{1, 2, ..., P −1}, it is
implicitly maintained due to the descending order of {λΓ,i}.
Speciﬁcally, the gradient of the objective (17) w.r.t. Mi is
g(Mi) := −
∂
∂Mi
Pmin(K,P )
i=1
3M 2
i λ2
˜Γ,i
3M 2
i +2η2 , which equals
gi(Mi) =
(
−
12Miη2λ2
Γ,i
(3M 2
i+2η2)2
i ≤K,
0
otherwise.
At the kth iteration, the gradient vector g(k) is thus given by
g(k) = [g1(M (k)
1
), g2(M (k)
2
), ..., gP (M (k)
P )]T .
(18)
The elements in (18) respectively represent the distortion
decreasing efﬁciency of the extra level we assign to each
quantizer at the kth iteration. The smallest entry of (18)
determines which quantizer is assigned an additional level,
repeating until the budget M is exhausted. The resulting
greedy allocation algorithm is summarized as Algorithm 1.
Once the bit assignment is obtained, the sampling operator and
its corresponding MSE are obtained using Proposition 1, while
the digital reconstruction ﬁlter is computed via Lemma 1.
2) Analysis of Algorithm 1: In the method summarized as
Algorithm 1, we use a greedy approach based on an MSE
measure achievable using real-valued assignments to optimize
the discrete bit allocation. Therefore, to assess the validity
5

## Page 6

Algorithm 1 Greedy bit allocation algorithm
Input: {λ˜Γ,i}
Output: Bit allocation {Mi}.
Initialize: k = 1 and set M (0)
i
= 1 for ∀i ∈P.
1: while PP
i=1 log2 M (k)
i
≤log2 M do
2:
Compute g(k) via (18);
3:
Update M (k+1)
i
= M (k)
i
+ 1i=arg minj gj(M (k)
j
);
4:
k = k + 1;
5: end while
6: return {M (k−1)
i
}
0
5
10
15
20
25
30
0.5
1
1.5
2
2.5
3
3.5
Objective value
MSE with integer assignment
MSE with real-valued assignment
(a) log2 M = 10
0
5
10
15
20
25
30
1.5
2
2.5
3
3.5
4
4.5
5
Objective value
MSE with integer assignment
MSE with real-valued assignment
(b) log2 M = 20
Fig. 2. 30 examples with randomly generated λΓ,i.
of this approach, we ﬁrst numerically compare the ability of
Algorithm 1 to approach the excess MSE in (17) using discrete
bit assignments. Then, we characterize a sufﬁcient condition
for which Algorithm 1 yields an identical bit allocation.
Finally, since the number of quantized samples P is inherently
a design parameter of the system, we derive the setting of P
which optimizes the MSE achievable using Algorithm 1.
Empirical Evaluation: Fig. 2 compares the MSE achieved
with the discrete bit assignment computed using Algorithm 1
to the MSE evaluated via (17) for 30 i.i.d. examples with
P = K = 10. As the purpose of this study is to evaluate the
ability of Algorithm 1 to approach (17), we consider a toy
example where the descending sequence {λ˜Γ,i} is randomly
generated with normal Gaussian distribution; more realistic
experiments corresponding to different forms of graph signals
are reported in our simulation study in Section V.
Observing Fig. 2, we note that when the overall bit budget is
tight, i.e., log2 M = 10 bits as in Fig. 2(a), there is a small gap
between the MSE achieved using the discrete assignment and
that which can reached given the ability to assign arbitrary
quantization levels. However, as the bit budget increases to
log2 M = 20, which is still a relatively tight budget (allow-
ing merely 2 bits per quantizer for a standard identical bit
assignment), we observe in Fig. 2(b) that the gap becomes
negligible. This study indicates that although Algorithm 1 is
designed based on an MSE measure typically not achievable
with integer bit assignments, it allows to approach it to within
a small gap, which effectively vanishes as the bit budget grows.
Identical
Bit
Assignment: As discussed in Subsec-
tion III-B, while our derivation relies on representing the joint
sampling and quantization of bandlimited graph signals as a
task-based quantization setup, the resulting formulation differs
from that studied in the context of ADC design in [21]. One
of the key differences between the graph signal compression
task considered here and the design of hybrid analog/digital
acquisition studied in [21] stems from the additional degree
of freedom in the ability to allocate different bit assignments
between the quantizers. As the restriction to utilize identical
quantizers may facilitate the design of the compression system,
we next identify a sufﬁcient and necessary condition for which
Algorithm 1 yields an identical bit allocation:
Corollary 1. Algorithm 1 yields an identical bit assignment
Mi ≡Ma := ⌊M 1/P ⌋for each i = 1, . . . , P where P ≤K
when the following conditions are satisﬁed:
(Ma)P ≤M < (Ma)P + (Ma)P −1,
(19a)
g1(Ma) < gP (Ma −1).
(19b)
Proof: The proof is given in Appendix C.
Corollary 1 identiﬁes conditions for which, when satisﬁed,
one can simply utilize conventional identical bit allocation
rather than go through the greedy optimization in Algorithm 1.
Nonetheless, the conditions (19) are not that commonly sat-
isﬁed by graph signals. The condition (19b) hints that the
λΓ,P should be close to λΓ,1. However, the smoothness of the
graph signal determines that the weight of the low-frequency
component is usually much greater than the weight of the
high-frequency component, i.e., σ1 ≫σP , which potentially
contradicts (19b). On the other hand, when (19a) is not
satisﬁed, the greedy-based algorithm we proposed can also
improve the performance by using the redundant bits.
Number of Quantized Samples: Algorithm 1 assigns the
overall bit budget among P quantized samples, where P is
ﬁxed. Nonetheless, it may yield an assignment Mi = 1,
implying that the ith quantizer has a single quantization level
and is thus inactive. For instance, since the gradients in (18)
equal zero for i > K and are strictly negative for i ≤K,
it follows that Algorithm 1 assigns zero bits (i.e., Mi = 1)
for quantizers of index i > K. While increasing P to be
larger than the spectral support K thus does not affect the
MSE as these quantizers will no be active, using less than
K quantized samples may result in different achievable MSE
values. Consequently, in the following corollary we identify
the number of quantized samples P which minimizes the MSE
without setting any quantizer to be inactive:
Corollary 2. The MSE minimizing number of actively quan-
tized samples P ∗satisﬁes lP ∗≤log2 M < lP ∗+1, where
li =
(
1 + Pi
j=1 log2⌈˜lj,i⌉,
i ≤K,
+∞,
i > K.
(20)
Here, ˜lj,i ∈R is given by the solution to the following equality:
gj(˜lj,i) = gi(1),
(21)
where gi(•) is i-th element in the gradient of (17) w.r.t. Mi.
Proof: The proof is given in Appendix D.
Corollary 2 provides an increasing sequence, which divides
the feasible interval of M decision levels into K sub-intervals.
Before designing the graph signal compression system, we can
determine the expected number of quantizers according to the
size of M. When the total number of bits is large enough
satisfying log2 M ≥lK, K samples are actively quantized;
When the total number of bits is limited, as discussed above,
6

## Page 7

some inactive quantizers can be removed. In such a case, the
reduction in the number of quantized samples also reduces
the computational complexity of Algorithm 1 since it needs to
compute P-length gradient vector in each iteration. However,
for a speciﬁc graph signal compression task, if the number of
samples that can be taken P is less than P ∗of Corollary 2,
then all samples should be quantized with at least two bits.
IV. JOINT SAMPLING AND QUANTIZATION USING
FREQUENCY-DOMAIN GRAPH FILTERS
Section III designs the compression rule without imposing
any constraints on the sampling matrix, i.e., setting Ψ = ISF,
thus allowing the graph ﬁlter F to be any N × N matrix. In
this section, we specialize our analysis to frequency-domain
graph ﬁlters. We ﬁrst present in Subsection IV-A an alternative
problem formulation, obtained by restricting the graph ﬁlter in
(P2) to implement frequency-domain graph ﬁltering. Based on
the modiﬁed problem formulation, we propose an algorithm
to tune the bit allocation and sampling set design in Subsec-
tion IV-B, after which we provide an alternating optimization
algorithm to obtain the overall compression scheme in Subsec-
tion IV-C, and discuss the resulting design in Subsection IV-D.
A. Problem Formulation
To formulate the compression problem using frequency-
domain graph ﬁlters, we return to (P2), which serves as a basis
for the design of the unconstrained system in Section III. The
resulting formulation is stated in the following lemma:
Lemma
2. When the sampling matrix Ψ is restricted
to represent frequency-domain graph ﬁltering, i.e., Ψ =
ISUF(Λ)UT , then, by deﬁning H ≜UF(Λ) ˜Λ2F(Λ)UT
and X ≜UF(Λ) ˜ΛF(Λ)UT , (P2) is specialized into
max
IS,F (Λ),{Mi}Tr

ISHIT
S
 ISXIT
S + G
−1
,
s.t.
P
X
i=1
log2 Mi ≤log2 M, Mi ∈Z+, i ∈P.
(P3)
Proof: The proof is given in Appendix E.
Problem (P3) specializes (P2) to sampling using frequency-
domain graph ﬁltering. It translates the compression system
design into the joint optimization of the frequency-domain
graph ﬁlter F = UF(Λ)UT , the sampling set selection
IS, and the bit allocation {Mi}. In particular, F(Λ) affects
the matrices H and X, while the bit setting is implicitly
encapsulated in the distortion matrix G (10). Problem (P3)
is non-convex due to the coupling its variables. To tackle this
challenging optimization, in the following subsections, we ﬁrst
show how one can tune IS and {Mi} for a given graph ﬁlter
F, based on which we propose an alternating optimization
algorithm to jointly set the compression system parameters.
B. Sampling Set and Bit Allocation Design
Here, we solve (P3) under a given ﬁlter F to obtain the
sampling set S and bit allocation {Mi}. While the number of
samples taken is |S| = P, the number of actively quantized
samples can be smaller, as revealed by Corollary 2. Conse-
quently, while the number of samples taken is P, which is
ﬁxed here, the optimization of the sampling set should account
for the fact that possibly less than P samples are actively
quantized. Note that this requirement was not encountered
when considering unconstrained graph ﬁlters, where Ψ can
be any P × N matrix, while here Ψ needs to be explicitly
divided into sampling set selection IS and the graph ﬁlter F.
To optimize the sampling set in light of the above consider-
ation, we allow S to contain only the actively quantized, i.e.,
|S| ≤P. To formulate the resulting problem, we deﬁne
˜
M i ≜
(
Mj
i = (S)j,
1
otherwise,
(22)
where (S)j is the j-th sampling vertex in S. Note that in (22)
there is a one-to-one correspondence between {Mj}, Sj and
{ ˜
M i}, i.e., {Mj} and S are uniquely determined by { ˜
M i}.
Therefore, by letting G{ ˜
Mj} be the matrix G in (10) with the
bit allocation n { ˜
Mj}, the joint optimization of the sampling
set and the bit allocation for a given F can be formulated as
max
{ ˜
Mi}
Tr

ISHIT
S

ISXIT
S + G{ ˜
Mi}
−1
,
s.t.
P
X
i=1
log2 ˜
Mi ≤log2 M, ˜
Mi ∈Z+, i ∈P,
(23)
S = {j| ˜
Mj > 1}, |S| ≤P.
To solve (23), we propose a greedy method starting with
˜
M
(0)
i
= 1 for each i ∈N ≜{1, 2, . . . , N} and select a
quantizer to assign an additional level, as in Algorithm 1. In
the k-th iteration, we deﬁne the sampling set and bit allocation
as S(k) and { ˜
M (k)
i
} respectively. The objective (23) is now
f({ ˜
M (k)
i
})≜Tr

IS(k)HIT
S(k)

IS(k)XIT
S(k) +G{ ˜
M (k)
i
}
−1
.
In each iteration, we select which ˜
Mi to increment by approx-
imating the derivative of f(·) w.r.t. each ˜
Mi. The derivative
w.r.t. the integer ˜
Mi is approximated by
∂f({ ˜
M (k)
i
})
∂˜
Mj
≈q(k)
j
≜f({ ˜
M (k)
i
+ 1i=j}) −f({ ˜
M (k)
i
}).
At the kth iteration, the approximated gradient vector q(k) is
∇{ ˜
M (k)
i
}f({ ˜
M (k)
i
}) ≈q(k) ≜[q(k)
1 , q(k)
2 , ..., q(k)
N ]T .
(24)
Our proposed greedy method iteratively updates { ˜
Mi} while
accounting for the constraints on the overall number of bits and
the maximal number of actively qauntized samples imposed
in (23). To that aim we deﬁne a set of possible sample
indices I, which is initialized to N, i.e., all possible samples.
In each iteartion, we choose one quantizer index in I to
assign an additional level, by selecting the one which we
expect to contribute most substantially to the objective in (23),
determined by the largest entry of (24). Since the number
of actively quantized samples cannot go beyond P, once P
different samples are actively quantized, we ﬁx the set I to
only consider the indexes of these samples subsequently. The
overall process is summarized in Algorithm 2.
7

## Page 8

Algorithm 2 Bit allocation for a given graph ﬁlter
Input: Graph ﬁlter matrix F, maximal number of samples P;
Output: Bit allocation Mi and Sampling set S;
Initialize: k = 0, bit allocation ˜
M (0)
i
= 1 for ∀i ∈N, possible
indices I = N, and sampling set S(0) = ∅.
1: while P
i log2 ˜
M (k)
i
≤log2 M
do
2:
Compute q(k)
i
for each i ∈I;
3:
Update ˜
M (k+1)
i
= ˜
M (k)
i
+ 1i=arg mini∈I q(k)
i ;
4:
if |S(k)| = P then
5:
Set possible indices I = S(k+1) = S(k);
6:
else
7:
Update S(k+1) = {i| ˜
M (k+1)
i
> 1};
8:
end if
9:
k = k + 1;
10: end while
11: return { ˜
Mi} = { ˜
M (k−1)
i
}, S = S(k−1).
C. Alternate Optimization Algorithm Design
We proceed to solve (P3) for a given bit allocation {Mi}
and sampling set S. In this case, (P3) is simpliﬁed to
max
F (Λ) Tr

ISHIT
S
 ISXIT
S + G
−1
.
(25)
The dependency of (25) on F(Λ) is encapsulated in H and X.
Recalling the deﬁnition of the frequency domain graph ﬁlter
in (Deﬁnition 3) and the imposed polynomial structure, i.e.,
F(Λ) = PK0
i=0 βiΛi, we aim to optimize (25) with respect
to the coefﬁcients {βi}. Since F(Λ) is diagonal, we do this
by ﬁrst optimizing its diagonal entries, denoted {˜λi}N
i=1, after
which we approximate them using a setof {βi}K0
i=1.
To solve (25) with respect to {˜λi} , we consider the alternate
optimization method. In particular, we optimize each ˜λi for
ﬁxed {˜λj}j̸=i, repeating and updating this process for each i
until convergence is achieved. Now, for a ﬁxed index i ∈N,
it follows from the deﬁnitions of X (see Lemma 2) and G
in (10) that ISXIT
S + G = ˜λ2
i B + C, where the |S| × |S|
matrices B and C are independent of ˜λi, and are deﬁned as
 B

p,q ≜mp,q
 σ2
i + σ2
(U)(S)p,i(U)(S)q,i,
 C

p,q ≜mp,q
N
X
j=1,j̸=i
˜λ2
j
 σ2
j + σ2
(U)(S)p,j(U)(S)q,j,
with mp,q ≜1 + 2η2
3M 2
p 1p=q. We can now write (and solve) the
optimization of (25) w.r.t. a single ˜λi as
max
˜λi
Tr

ISHIT
S

˜λiB + C
−1
.
(26)
Lemma 3. The solution to (26) is ˜λ∗
i =
q
ˆλ∗
i , where
ˆλ∗
i = min
ˆλi≥0
P
X
j=1
aj
ˆλi + (Λx)j,j
.
(27)
In (27), we deﬁne ai ≜(A)2
i,i(Λx)j,j −PN
n=1,n̸=i ˆλn(A)2
i,j,
where A ≜UT
x B−1/2ISU ˜Λ, with Ux and Λx being the
eigenvectors and eigenvalues of B−1/2CB−1/2, respectively.
Proof. The proof is given in Appendix F.
Problem (27) is a concave-convex fractional programming
problem, which can be efﬁciently solved using the quadratic
transform [34]. After ˜λ∗
i is obtained, we update the value
of F(Λ) and optimize the remaining ˜λi. This procedure is
repeated until a pre-speciﬁed convergence condition ϵ is met,
or a maximal number of iterations T is reached. The resulting
alternating algorithm for constrained graph ﬁlter design is
summarized in Algorithm
3, where ϵ is the threshold and
T is the maximum number of iterations.
Algorithm 3 Constrained graph ﬁlter for compression design
Input: Sampling set S and bit assignment {Mi};
Output: Graph ﬁlter matrix F;
Initialize: ˜λ(0)
i
= 1, i ∈N.
1: for k = 1, 2, . . . , T do
2:
for i ∈S do
3:
Update ˜λ(k)
i
= ˜λ∗
i computed via Lemma 3;
4:
end for
5:
if
[˜λ(k)
1 , . . . , ˜λ(k)
N ] −[˜λ(k−1)
1
, . . . , ˜λ(k−1)
N
]
2
2 ≤ϵ then
6:
Break;
7:
end if
8: end for
9: Compute F(Λ) = diag{˜λ(k−1)
1
, ˜λ(k−1)
2
, ..., ˜λ(k−1)
N
};
10: return F = UF(Λ)UT .
Finally, the overall alternating optimization based algorithm
for setting the frequency domain graph ﬁlter, sampling set, and
bit allocation based on (P3) is summarized as Algorithm 4.
The algorithm alternates between setting the frequency-domain
graph ﬁlter for ﬁxed sampling set and bit allocation via Al-
gorithm 3, and optimizing the sampling set and bit allocation
for ﬁxed graph ﬁlter via Algorithm 2.
Algorithm 4 Compression system design with frequency
domain graph ﬁlters
Input: Graph Fourier basis matrix UT , maximal samples P;
Output: Graph ﬁlter F, bit allocation {Mi}, sampling set S;
Initialize: F(0) = I, k = 0.
1: repeat
2:
Compute S(k) and {M (k)
i
} for F(k) via Algorithm 2;
3:
Update F(k+1) for S(k) and {Mi}(k) via Algorithm 3;
4:
k = k + 1;
5: until ∥F(k) −F(k−1)∥≤ϵ
6: return F = F(k), {Mi} = {M (k−1)
i
}, S = S(k−1) .
D. Discussion
Our proposed compression scheme is based on jointly
designing the sampling and quantization mappings to achieve
a ﬁnite bit representation of the graph signal in a manner
which allows its accurate reconstruction. This is achieved
utilizing a sampling matrix which, instead of selecting a
subset of the nodes as in [6], samples linear combinations
of the graph signal elements using frequency domain graph
ﬁltering. The increased ﬂexibility in the design of the sampling
mapping is exploited to generate a sampling set such that the
graph signal spectrum can be accurately recovered from the
8

## Page 9

20
30
40
50
60
70
80
90
100
log2Mbits
0
0.02
0.04
0.06
0.08
0.1
0.12
0.14
MSE
Joint sampling and quantization, non-identical quantizers (Algorithm 1)
Separate sampling and quantization, non-identical quantizers
No-quantization (MMSE)
Joint sampling and quantization, identical quantizers
Joint sampling and quantization, non-identical quantizers (Algorithm 4)
78
80
82
-2
0
2
4
6
10-3
Fig. 3. MSE versus the number of bits for different schemes.
quantized samples. In particular, we do this by accounting
for the statistical moments of the noisy graph signal via the
relaxed formulation (P3) when jointly tuning the sampling
matrix and the quantization rule. We focus here on implement-
ing such compression mechanisms using either unconstrained
graph ﬁlters as well as structured frequency domain graph
ﬁlters. However, one can consider additional forms of graph
ﬁltering which may be preferable in some applications, e.g.,
vertex domain graph ﬁlters [8]. Nonetheless, we leave the
specialization of the proposed joint sampling and quantization
mechanism to alternative forms of graph ﬁlters for future work.
Finally, we note that our design considers a limit on the
overall number of bits used by the quantizers to generate the
ﬁnite bit representation. A less distorted ﬁnite bit representa-
tion can be obtained by replacing the uniform scalar quantizers
with vector quantization [32, Ch. 23], though such mappings
tend to be highly complex, as opposed to simpliﬁed uniform
quantization. An alternative approach to improve upon the
existing mechanism is to further compress Q(Ψx) via lossless
source coding, e.g., entropy coding [35, Ch. 13], to Q(Ψx), as
done in [36]. Doing so results in an equivalent representation
whose number of bits is dictated by the entropy of Q(Ψx)
(which is not larger than log2 M). This implies that the re-
sulting ﬁnite bit representation can be further compressed, and
motivates extending our analysis to constrain the entropy of
Q(Ψx) rather than its bit budget, i.e., via entropy-constrained
quantization [15], which we leave for future research.
V. NUMERICAL EVALUATIONS
In this section, we evaluate the performance of the proposed
joint sampling and quantization methods for graph signal
compression.1 In Subsection V-A, we simulate graph signals
representing measurements taken using parameters provided
in GSP toolbox [37]. We then use the proposed scheme to
compress graph signals representing real-world temperature
data in Subsection V-B. Finally, in Subsection V-C, we apply
the proposed sampling algorithm to image compression.
Throughput this section, we evaluate the MSE achieved
when compressing the graph signal using to the following
schemes: (1) Joint sampling and quantization with uncon-
strained ﬁlters via Algorithm 1; (2) Separate sampling and
quantization with non-identical quantizers as in [20]; (3)
the MMSE estimate achievable of Ukc which is achievable
1The source code used in this section is available online in the following
link: https://github.com/Pei65536/Graph Signal Compression.git
-30
-29
-28
-27
-26
-25
-24
-23
-22
-21
-20
Power of noise (dB)
0
0.01
0.02
0.03
0.04
0.05
0.06
0.07
0.08
0.09
MSE
Joint sampling and quantization, non-identical quantizers (Algorithm 1)
Separate sampling and quantization, non-identical quantizers
No-quantization (MMSE)
Joint sampling and quantization, identical quantizers
Joint sampling and quantization, non-identical quantizers (Algorithm 4)
Fig. 4. MSE versus the variance of noise.
with inﬁnite resolution quantization, i.e., without quantization
constraints; (4) Joint sampling and quantization with identical
quantizers as in [21]; and (5) Joint sampling and quantization
with frequency domain graph ﬁlters via Algorithm 4.
A. Synthetic Data
We begin with synthetic simulated scenario. Here, the graph
signals represent measurements acquired by a sensor network
comprised of N = 100 nodes, where each node is connected
with its neighbors located within its transmission range, and
the bandwidth is K = 20. We then take 1000 different noisy
bandlimited graph signals with zero mean and covariance Cx,
in which the non-zero GFT coefﬁcients are randomized from
N(0, 1/λi), where λi is the ith eigenvalue of L.
We ﬁrst numerically evaluate the MSE in compressing the
graph signals versus the bit budget log2 M, which takes values
in [20, 120], while setting the Gaussian noise power to σ2
0 =
−30 dB. The results, depicted in Fig. 3. Observing Fig. 3,
we note that Algorithm 1 achieves better performance than
using separately designed sampling and quantization as well
as applying the joint design with identical quantizers of [21].
The separate sampling and quantization mechanism of [20],
in which the sampling operation is carried out by mere node
selection without accounting for the underlying statistics of
the signal, achieves the highest MSE values here, while the
proposed schemes is within a small MSE gap of less than 0.02
from the MMSE for bit budgets as small as log2 M = 40. We
also noted that Algorithm 4 allows achieving MSE which is
comparable to that achieved using unconstrained graph ﬁlters
for bit budgets satisfying log2 M ≥40.
Next, we numerically evaluate the effect of the additive
noise power on the compression accuracy. To that aim, we
compare the achievable MSEs of the schemes simulated in
Fig. 3, while ﬁxing the bit budget to be log2 M = 60, and
letting the noise variance σ2
0 grow from −30 dB to −20 dB.
The results, depicted in Fig. 4, demonstrate that our proposed
schemes maintains their improved MSE over the competing
schemes for different levels of the noise variance. Our scheme
is shown to be notably more robust to the presence of noise
compared to the all the reference techniques, except for the
MMSE estimate, which is not constrained to any bit budget.
To visualize how the observed MSE gains are translated
into a more faithful recovery of compressed graph signals,
we depict a realization of a graph signal along with its
reconstruction using scheme (1) and the benchmarks (2) and
9

## Page 10

Original Graph Signal
-2.5
-2
-1.5
-1
-0.5
0
0.5
1
1.5
2
(a)
Joint sampling and quantization, non-identical quantizers (Algorithm 1)
-2.5
-2
-1.5
-1
-0.5
0
0.5
1
1.5
2
(b)
Joint sampling and quantization, identical quantizers
-2.5
-2
-1.5
-1
-0.5
0
0.5
1
1.5
2
(c)
Separate sampling and quantization, non-identical quantizers
-2
-1.5
-1
-0.5
0
0.5
1
1.5
2
2.5
(d)
Fig. 5. Synthetic graph signal recovery experiments: (a) Graph signal; (b)
Reconstructed signal via Algorithm 1; (c) Reconstructed signal with separate
sampling and quantization with non-identical quantizers; (d) Reconstructed
signal with joint sampling and quantization with identical quantizers.
(4) in Fig. 5. Here, the noise level is set to σ2
0 = −30dB, while
the overall bit budget is log2 M = 60.Fig. 5 demonstrates that
the proposed scheme based on joint sampling and quantization
allows to compress the graph signal into a compact bit rep-
resentation while resulting in recovered graph ﬁlters within a
close match of the original graph signal. When using identical
quantizers as in [21], the quality of the recovery signal in
Fig. 5(d) is degraded and not as smooth as it is in Fig, 5(b).
This follows since the low frequency components usually
occupy a relatively large range, and using identical quantizers
limits the recovery of the low frequency components. For
the separate sampling and quantization scheme of [20], we
observe in Fig. 5(c) poor recovery on some vertexes, which is
caused by the ampliﬁcation of quantization noise and additive
Gaussian noise in restoring unsampled vertexes.
B. Non-Synthetic Graph Signals
In order to further illustrate the practicability of our graph
signal compression algorithms, we apply the proposed Algo-
rithm 1 and Algorithm 4 to compress non-synthetic graph
signals representing temperature data2The data is obtained
from the China Meteorological Data Center, representing
measurements taken in January of each year from 2018 to 2021
via China’s meteorological sensors whose distribution is is
shown in Fig. 6 (a). The location of each sensor is determined
by its latitude and longitude. It is noted that temperature data
distribution depends not only on the geographical location, but
is also affected by the surrounding environment and human
actions. Therefore, we construct the graph structure for these
measurements based on the smoothness of the data, rather than
directly constructing the graph structure based on the distance
threshold, e.g., as in K-nearest neighbours construction. An
example of a resulting graph signal is illustrated in Fig. 6
(b). For the adopted data, the temperature ﬂuctuation tends to
2The dateset could be downloaded in the following link: http://data.cma.cn
-3
-2
-1
0
1
2
3
106
2
2.5
3
3.5
4
4.5
5
5.5
6
106
China ground international exhchange station
(a)
(b)
Fig. 6. (a) Geographical location of China ground sensors. (b) A temperature
example in formulated graph structure.
a stable state. The statistical information of the temperature
data shows that when the bandwidth is greater than 10, the
frequency domain component is close to 0. Therefore, we
assume that the bandwidth of the data is K = 10.
The achievable reconstruction MSE for these non-synthetic
graph signals versus the overall number of bits is depicted
in Fig. 7. It is noted that in this scenario there is no noise
signal that is manually introduced, and the equivalent noise
is that which stems from the error in approximating the
graph signal as have bandwidth of K = 10. We observe
in Fig. 7 that the MSE trends versus the total number of
bits is preserved as in the synthetic case reported in the
previous subsection. Speciﬁcally, it is found that compression
with unconstrained graph ﬁlters via Algorithm 1 outperforms
all considered bit-constrained benchmarks, and that similar
performance is achieved with frequency domain graph ﬁlters
for at least 20 bits using Algorithm 4.
C. Application in Image Compression
GSP is not only applied to the data with irregular structures,
but can also be used to represent l signals such as images,
video signals, etc. Here, we apply our proposed Algorithm 1
to image compression. Due to the self-similarity in images, the
same or similar structures are likely to recur throughout. For
simplicity, we apply regular line and grid graph topologies,
but with unequal edge weights that can adapt to the speciﬁc
characteristic of an image or a set of images [38].
In particular, we use the classic ‘Lena‘ image comprised
512 × 512 pixels, which are represented as a concatenation
from 16384 graph signals in 4 × 4 patches. For each graph
signal, we approximate its bandwidth to be K = 4, i.e.,
we discard high-frequency components, which is relatively
quite small. The statistical moments used in our derivation are
obtained by averaging over the patches of the image. We com-
pare the performance of three image compression schemes: 1)
The Discrete Cosine Transform (DCT); 2) Separate sampling
with quantization with non-identical quantizers [20]; 3) The
proposed Algorithm 1. For consistency, we set the block size
of DCT to be 4 × 4. We set log2 M = 64 for each GFT block
and DCT block. The results, depicted in Fig. 8, indicate that
the proposed mechanism for joint sampling and quantization
which considers graph signals can also be beneﬁcial for image
compression, resulting in a faithful recovery of natural images
compressed into a low bit representation.
VI. CONCLUSIONS
In this work, we studied the compression of bandlimited
graph signals into a ﬁnite-length sequence of bits by joint
10

## Page 11

10
15
20
25
30
35
40
45
50
55
60
log2 M bit
0
5
10
15
20
25
30
35
40
MSE
Joint sampling and quantization, non-identical quantizers (Algorithm 1)
Separate sampling and quantization, non-identical quantizers
No-quantization (MMSE)
Joint sampling and quantization, identical quantizers
Joint sampling and quantization, non-identical quantizers (Algorithm 4)
Fig. 7. MSE versus the number of bits.
sampling and quantization. Our derivation is based on the
identiﬁed similarity between graph signal compression and
task-based quantization, which is a framework for conﬁguring
ADCs with analog pre-processing. We formulated the design
problem, which is in general shown to be non-convex. We then
presented a relaxed formulation considering linear recovery
with non-overloaded quantizers. The relaxed problem was used
to design sampling mechanisms for a given allocation of the
available bit budget among the quantizers, which in turn is
used to derive a greedy bit allocation algorithm. Next, we
proposed a sampling operator as a row-selection matrix with a
graph ﬁlter, restricted to combining elements corresponding to
the neighbouring nodes of the graph. We applied the proposed
algorithm to both synthetic and non-synthetic data. Numeri-
cal results show that the proposed scheme compresses high
dimensional graph signals into a limited amount of bits while
allowing their recovery with an error within a small gap from
the MMSE, achievable with inﬁnite resolution quantziation.
APPENDIX
A. Proof of Proposition 1
We ﬁrst note that the MSE achievable using the linear
recovery matrix Φ∗in Lemma 1 is given by
MSE(Ψ) = Tr
 Γ∗CxΓ∗T 
−Tr

Γ∗CxΨT  ΨCxΨT + G
−1 ΨCxΓ∗T 
.
(A.1)
By discarding the constant term Tr
 Γ∗CxΓ∗T 
in (A.1), the
matrix Ψ∗that minimizes the MSE can be obtained via
ˆΨ
∗= arg max
ˆΨ Tr

ˆΓ ˆΨ
T 
ˆΨ ˆΨ
T + I
−1 ˆΨˆΓ
T 
,
(A.2)
where
ˆΨ
≜=
G−1/2ΨC1/2
x , ˆΓ
≜=
Γ∗C1/2
x . Now, by
writing the singular value decomposition ˆΨ = UΨΞΨVT
Ψ,
and recalling the deﬁnition of G, we obtain that Gi,i3M 2
i
2η2
=
 ΨCxΨT 
i,i =
 G1/2 ˆΨ ˆΨT G1/2
i,i, which results in
(UΨΞ2
ΨUΨ
T )i,i = 3M 2
i
2η2 .
(A.3)
According to majorization theory [39], (A.3) is satisﬁed if
and only if {3M 2
i /2η2} ≺{αi}, where αi = (ΞΨ)2
i,i. Then
problem (A.2) is transformed into
max
VΨ,ΞΨ Tr

VT
ΨˆΓ
T ˆΓVΨΞT
Ψ
 ΞΨΞT
Ψ + I
−1ΞΨ

,
s.t.
3M 2
i
2η2

≺{αi} , αi = (ΞΨ)2
i,i.
(A.4)
(a)
(b)
(c)
(d)
Fig. 8.
Image compression. (a) The original image; (b) The compressed
image via DCT (c) The compressed image via joint sampling and quantization
with identical quantizers; (d) The compressed image via Algorithm 2.
We note that ΞT
Ψ
 ΞΨΞT
Ψ + I
−1ΞΨ is a diagonal matrix
with diagonal elements
αi
αi+1, i ∈P, which are a descending
sequence since {αi} is arranged in descending order. Then,
the optimal VΨ should be the right singular vectors matrix
of ˆΓ. By substituting (2) and (4) into the deﬁnition of ˆΓ,
we obtain ˆΓ =
ˆΛ( ˜Λ−1)KUT . Note that ˆΛ( ˜Λ−1/2)K is
a diagonal matrix with descending diagonal elements, i.e.,
VT
Ψ = UT . Consequently, ˜Γ = VT
ΨˆΓ
T ˆΓVΨ is a diagonal
matrix, whose diagonal elements arrange in a descending
order. The remaining optimization of (A.4) is thus
max
αi
P
X
i=1
λ2
Γ,iαi
αi + 1,
(A.5)
s.t.
P
X
i=1
αi =
P
X
i=1
3M 2
i
2η2 ,
p
X
i=1
αi ≥
p
X
i=1
3M 2
i
2η2 , ∀p ∈P,
where λΓ,i is the i-th singular value of ˆΓ. The constraints in
(A.5) are the expanded form of
n
3M 2
i
2η2
o
≺w {αi}. We note
that (A.5) is convex. Once {αi} are obtained, the resulting
sampling matrix is Ψ∗= G1/2UΨΞΨVT
ΨC−1/2
x
. While the
entries of G depend on Ψ, we can still obtain the sampling
matrix by letting gi = G1/2
i,i , and substituting the expression of
Ψ∗into the deﬁnition of G. This results in
2η2(ΨCxΨT)i,i
3M 2
i
=
g2
i , and thus
 G1/2UΨΞ2
ΨUΨ
T G1/2
i,i =
g2
i 3M 2
i
2η2 , which
holds for any gi by (A.3). We can thus compute Ψ with G = I,
obtaining Ψ∗= UΨΞΨ ˜Λ
−1/2UT , concluding the proof.
B. Proof of Theorem 1
Lemma 1 characterizes the MSE minimizing sampling op-
erator Ψ for ﬁxed bit allocation {Mi}. To optimize {Mi}, we
focus on the case where {Mi} are not limited to be integer,
resulting in the following formulation
11

## Page 12

max
{Mi},{αi}
P
X
i=1
λ2
Γ,iαi
αi + 1,
(B.1)
s.t.
3M 2
i
2η2

≺{αi},
P
X
i=1
log2 Mi ≤log2 M, Mi ≥1.
Problem (B.1) is non-convex and difﬁcult to solve, Hence, we
ﬁrst propose the following lemma to simplify (B.1).
Lemma B.1. The solution of (B.1) satisﬁes 3M 2
i
2η2 =αi, ∀i ∈P.
Proof. We ﬁrst recall the following result from [39]. Let φ :
Dn →R be a real-valued function continuous on Dn ≜{x ∈
Rn : x1 ≥... ≥xn} and continuously differentiable on the
interior of Dn. Then φ is Schur-convex (Schur-concave) on Dn
if and only if ∂φ(x)
∂xi
is decreasing (increasing) in i = 1, ..., n.
Assume that
 ´
Mi
	
is the optimal bit allocation and

´αi
	
is
the corresponding diagonal elements satisfying {3 ´
M 2
i /2η2} ≺
{´αi}. Denote M (1) = PP
i=1 log2 3 ´
M 2
i /2η2 and M (2) =
PP
i=1 ´αi. Since P log2 Mi is Schur-concave with respect to
Mi, then M (1) ≥M (2). Thus, one can propose another
 `
Mi
	
and

`αi
	
satisfying 3 `
M 2
i /2η2 = `αi = ´ακ0
i , where
κ0 = M (1)/M (2) ≥1. Clearly,
 `
Mi
	
,

`αi
	
are feasible for
(B.1), with better MSE than

´αi
	
, proving the lemma.
By Proposition 1, the sampling matrix should be
Ψ∗= UΨΞΨ ˜Λ
−1/2UT
(B.2)
Substituting Lemma B.1 into (B.2), the matrix UΨ becomes
the identity. Hence, the sampling matrix would be a diagonal
matrix multiplied by the unitary UT . The proof of Proposi-
tion 1 reveals that the diagonal matrix G−1/2 which is leftmost
in the expression for Ψ∗could be arbitrarily set. As a result,
the MSE is minimized by the sampling matrix Ψ∗= UT
K.
To
determine
{Mi},
we
write
problem
(B.1)
as
max
{Mi}
PP
i=1
λ2
Γ,i3M 2
i
3M 2
i +2η2 , s.t. PP
i=1 log2 Mi ≤log2 M, Mi ≥
1, Mi
≥
Mi+1, which is still non-convex. To tackle
this, we obtain a local solution and prove its optimality.
According to Lagrange Duality theorem,we introduce the
multipliers β∗∈R and ν∗∈RP . While this procedure
does not impose monotonicity on {Mi}, it is implicitly
maintained since λΓ,i are arranged in a descending order.
For simplicity, we denote mi = M 2
i , and write the KKT
constraints as PP
i=1 log2 m∗
i
≤2 log2 M, with m∗
i
≥1,
ν∗
i (m∗
i −1) = 0 and
6λ2
Γ,iη2
(3mi+2η2)2 −ν∗
i −β∗
mi = 0, for each
i ∈P. Eliminating the slack variable ν∗
i , we obtain that
  β∗
mi −
6λ2
Γ,iη2
(3mi+2η2)2

(m∗
i −1) = 0 should hold.
We focus on the equation
β∗
mi −
6λ2
Γ,iη2
(3mi+2η2)2 = 0, which
is solvable if and only if β∗≤λ2
Γ,i/4. The local optimal
mi is obtained here by (16). The optimal bit allocation is
expressed by the dual coefﬁcient β∗, which satisﬁes the sum
bits constraint f(β) = PP
i=1 log2 m∗
i ≤2 log2 M. Similar
as the proof of lemma B.1, we obtain that f(β) = 2 log2 M
should be satisﬁed. Further, f(β) is a decreasing function with
respect to β, i.e., the optimal β∗is the only one.
C. Proof of Corollary 1
We ﬁrst focus on the necessary condition. Due to the def-
inition of Ma, (Ma)P :=
 ⌊M 1/P ⌋
P ≤M, and M ≥M P
a
should be obviously satisﬁed. We then assume that M ≥
M P
a +M P −1
a
, there exists another scheme of bit assignment as
[Ma+1, Ma, . . . , Ma] obtained by Algorithm 1, which against
the previous assumption. The range of M is thus determined,
which corresponds to the necessity of (19a).
We note that λ˜Γ,i is a descending sequence, thus gi(Mi) ≥
gj(Mj) for Mi = Mj, and gi(Mi) > gi(Mi + 1) for Mi ≥1.
When Mi = Mj = Ma for each i ∈P, we deﬁne k0 =
(Ma)P −1, as we discussed above, the state M (k0)
i
should be
[Ma, Ma, . . . , Ma] and (k0+1)-th iteration should select P-th
quantizer rather than others, which means that gP (Ma −1) >
gi(Ma) for each i ∈P, and then simpliﬁed as gP (Ma −1) >
g1(Ma). The necessity of (19b) is thus proven.
For the sufﬁcient case, we assume both (19a) and (19b) are
satisﬁed. Greedy-based method is applied in Algorithm 1, we
try to describe the bit assignment process. In k-th iteration,
the particular index is selected by e = arg mini gi(M (k)
i
).
For example, we assume that in k1-th iteration, the state of
M (k1−1)
1
change from Ma −1 to Ma, i.e., M (k1)
1
= Ma. Here
the other bit assignment should satisfy that Mi(k1) < Ma
for each i = 2, . . . , P since gi(Mi) ≥gj(Mj) for Mi = Mj.
When (19b) is satisﬁed, it holds that g1(Ma) < gP (Ma−1) <
gP −1(Ma −1) · · · < g2(Ma −1). This, before the state of
MP change from Ma −1 to Ma, the state of M1 would not
be changed. Further, we assume that in kP -th iteration, the
state of M (kP −1)
P
change from Ma −1 to Ma, i.e., M (kP )
P
=
Ma. Consequently, gP −1(Ma) < gP −2(Ma) · · · < g1(Ma) <
gP (Ma −1) As a result, after the kP -th iteration, Mi = Ma
for each i = 1, . . . , P. Since (19a) limit the upper bound of
M, the greedy-based algorithm would be ended here due to
the termination condition, concluding the proof.
D. Proof of Corollary 2
We ﬁrst focus on the case when sum of bits is limited, which
may lead to P < K. We would complete this proof in two
parts, in which we separately propose that
P < i if
log2 M < li + 1,
(PartI)
P ≥i if
log2 M ≥li + 1.
(PartII)
Deﬁne P ∗as the number of quantizers obtained by Algo-
rithm 1, and its bit allocation as {M ∗
1 , M ∗
2 , ...., M ∗
P ∗}. For any
P ≤P ∗, we assume that P-th quantizer is added in the k0-th
iteration, i.e., P = arg maxk0 g(k0), thus
gP (1) ≥gj(M (k0)
j
) ≥gj(M ∗
j ), ∀j < P.
(D.1)
For the proof of the (PartI), we assume that P ≥i and
log2 M
< li + 1. Then, we deﬁne the bit allocation as
{ ˙M 1, ˙M 2, ..., ˙M P }, which should satisfy gi( ˙M i) ≤gP (1).
It follows that
˙M i ≥⌈˜lj⌉. Then we obtain PP
i=1 ˙M i ≥
PP −1
i=1
˙M i + 1 ≥li + 1, which contradicts the assumption.
To prove (PartII), consider the k1th iteration, where
gi(M (k)
i
) ≥gP (1) and gi(M (k)
i
+ 1) ≤gP (1) for i < P.
Note that M (k)
i
≤˜lj ≤M (k)
i
+ 1 and ⌈˜lj⌉= M (k)
i
+ 1. Thus,
for the (k1 + P −n)th iteration, where 1 ≤n ≤P −1,
12

## Page 13

P = arg maxk g(k) is obtained, proving (PartII). Further,
when log2 M ≥lK, the Kth quantizer is added, and there
is no (K + 1)th quantizer since the gradient vector satisﬁes
gi(Mi) = 0 for i > K. This results in (20).
E. Proof of Lemma 2
To prove the equivalence between (P2) and (P3), we
ﬁrst note that that Lemma 1, which characterizes the MSE
minimizing digital recovery ﬁlter, holds for any sampling
matrix Ψ, including those representing frequency-domain
graph ﬁltering. Consequently, by substituting the resulting
recovery matrix Φ = Γ∗CxΨT  ΨCxΨT + G
−1, we ar-
rive at
max
Ψ,{Mi} Tr

Γ∗CxΨT  ΨCxΨT + G
−1 ΨCxΓ∗T 
,
s.t. PP
i=1 log2 Mi ≤log2 M, Mi ∈Z+. By setting Ψ =
ISUF(Λ)UT to represent frequency-domain ﬁltering, we
transform the objective into (P3), proving the lemma.
F. Proof of Lemma 3
We ﬁrst transform (˜λ2
i B + C)−1 into
(˜λ2
i B + C)−1 = B−1/2
˜λ2
i I + B−1/2CB−1/2−1
B−1/2
(a)
= B−1/2
˜λ2
i I + UxΛxUT
x
−1
B−1/2
= B−1/2Ux

˜λ2
i I + Λx
−1
UT
x B−1/2, (F.1)
where (a) follows from the eigenvalue decomposition, where
Ux is a unitary matrix since both B and C are symmetric.
Substituting (F.1) and the deﬁnition of A into (26) yields
max
˜λi
Tr

A ˜Λ
2F(Λ)2AT 
˜λ2
i I + Λx
−1
.
(F.2)
Noting that ˜λi is in form of ˜λ2
i in (F.2), thus we deﬁne
ˆλi = ˜λ2
i , simplifying (F.2) into: max
ˆλi≥0
PP
j=1
PN
n=1 ˆλn(A)2
j,n
ˆλi+(Λx)j,j
.
Substituting the deﬁnition of {ai} yields (27).
REFERENCES
[1] P. Li, N. Shlezinger, H. Zhang, B. Wang, and Y. C. Eldar, “Graph signal
compression via task-based quantization,” in Proc. IEEE ICASSP, 2021.
[2] D. I. Shuman, S. K. Narang, P. Frossard, A. Ortega, and P. Van-
dergheynst, “The emerging ﬁeld of signal processing on graphs: Ex-
tending high-dimensional data analysis to networks and other irregular
domains,” IEEE Signal Process. Mag., vol. 30, no. 3, pp. 83–98, 2013.
[3] A. Sandryhaila and J. M. F. Moura, “Big data analysis with signal
processing on graphs: Representation and processing of massive data
sets with irregular structure,” IEEE Signal Process. Mag., vol. 31, no. 5,
pp. 80–90, 2014.
[4] ——, “Discrete signal processing on graphs,” IEEE Trans. Signal
Process., vol. 61, no. 7, pp. 1644–1656, 2013.
[5] A. Ortega, P. Frossard, J. Kovaˇcevi´c, J. M. F. Moura, and P. Van-
dergheynst, “Graph signal processing: Overview, challenges, and ap-
plications,” Proc. IEEE, vol. 106, no. 5, pp. 808–828, 2018.
[6] S. Chen, R. Varma, A. Sandryhaila, and J. Kovaˇcevi´c, “Discrete signal
processing on graphs: Sampling theory,” IEEE Trans. Signal Process.,
vol. 63, no. 24, pp. 6510–6523, 2015.
[7] Y. Tanaka and Y. C. Eldar, “Generalized sampling on graphs with
subspace and smoothness priors,” IEEE Trans. Signal Process., vol. 68,
pp. 2272–2286, 2020.
[8] Y. Tanaka, Y. C. Eldar, A. Ortega, and G. Cheung, “Sampling on graphs:
From theory to applications,” IEEE Signal Process. Mag., vol. 37, no. 6,
pp. 14–30, 2020.
[9] A. Sandryhaila and J. M. F. Moura, “Discrete signal processing on
graphs: Frequency analysis,” IEEE Trans. Signal Process., vol. 62,
no. 12, pp. 3042–3054, 2014.
[10] A. C. Ya˘gan and M. T. ¨Ozgen, “Spectral graph based vertex-frequency
wiener ﬁltering for image and graph signal denoising,” vol. 6, pp. 226–
240, 2020.
[11] A. Heimowitz and Y. C. Eldar, “A Markov variation approach to smooth
graph signal interpolation,” arXiv preprint arXiv:1806.03174, 2018.
[12] X. Dong, D. Thanou, P. Frossard, and P. Vandergheynst, “Learning
laplacian matrix in smooth graph signal representations,” IEEE Trans.
Signal Process., vol. 64, no. 23, pp. 6160–6173, 2016.
[13] S. P. Chepuri, S. Liu, G. Leus, and A. O. Hero, “Learning sparse graphs
under smoothness prior,” in Proc. IEEE ICASSP, 2017.
[14] L. F. O. Chamon and A. Ribeiro, “Greedy sampling of graph signals,”
IEEE Trans. Signal Process., vol. 66, no. 1, pp. 34–47, 2018.
[15] R. M. Gray and D. L. Neuhoff, “Quantization,” IEEE Trans. Inf. Theory,
vol. 44, no. 6, pp. 2325–2383, 1998.
[16] L. F. O. Chamon and A. Ribeiro, “Finite-precision effects on graph
ﬁlters,” in Proc. IEEE GlobalSIP, 2017, pp. 603–607.
[17] I. C. M. Nobre and P. Frossard, “Optimized quantization in distributed
graph signal processing,” in Proc. IEEE ICASSP, 2019.
[18] L. B. Saad, E. Isuﬁ, and B. Beferull-Lozano, “Graph ﬁltering with quan-
tization over random time-varying graphs,” in Proc. IEEE GlobalSIP,
2019.
[19] P. Di Lorenzo, S. Barbarossa, and P. Banelli, “Optimal power and bit
allocation for graph signal interpolation,” in Proc. IEEE ICASSP, 2018.
[20] Y. H. Kim and A. Ortega, “Toward optimal rate allocation to sampling
sets for bandlimited graph signals,” IEEE Signal Process. Lett., vol. 26,
no. 9, pp. 1364–1368, 2019.
[21] N. Shlezinger, Y. C. Eldar, and M. R. D. Rodrigues, “Hardware-limited
task-based quantization,” IEEE Trans. Signal Process., vol. 67, no. 20,
pp. 5223–5238, 2019.
[22] N. Shlezinger, Y. C. Eldar, and M. R. Rodrigues, “Asymptotic task-based
quantization with application to massive MIMO,” IEEE Trans. Signal
Process., vol. 67, no. 15, pp. 3995–4012, 2019.
[23] S. Salamtian, N. Shlezinger, Y. C. Eldar, and M. Medard, “Task-based
quantization for recovering quadratic functions using principal inertia
components,” in Proc. IEEE ISIT, 2019.
[24] P. Neuhaus, N. Shlezinger, M. D¨orpinghaus, Y. C. Eldar, and G. Fettweis,
“Task-based analog-to-digital converters,” IEEE Trans. Signal Process.,
early access, 2021.
[25] N. Shlezinger and Y. C. Eldar, “Task-based quantization with application
to MIMO receivers,” arXiv preprint arXiv:2002.04290, 2020.
[26] ——, “Deep task-based quantization,” Entropy, vol. 23, no. 1, p. 104,
2021.
[27] F. Gama, E. Isuﬁ, G. Leus, and A. Ribeiro, “Graphs, convolutions, and
neural networks: From graph ﬁlters to graph neural networks,” IEEE
Signal Process. Mag., vol. 37, no. 6, pp. 128–138, 2020.
[28] X. Dong, D. Thanou, P. Frossard, and P. Vandergheynst, “Learning
laplacian matrix in smooth graph signal representations,” IEEE Trans.
Signal Process., vol. 64, no. 23, pp. 6160–6173, 2016.
[29] D. K. Hammond, P. Vandergheynst, and R. Gribonval, “Wavelets on
graphs via spectral graph theory,” Applied and Computational Harmonic
Analysis, vol. 30, no. 2, pp. 129–150, 2011.
[30] R. M. Gray and T. G. Stockham, “Dithered quantizers,” IEEE Trans.
Inf. Theory, vol. 39, no. 3, pp. 805–812, 1993.
[31] B. Widrow, I. Kollar, and M.-C. Liu, “Statistical theory of quantization,”
IEEE Trans. Instrum. Meas., vol. 45, no. 2, pp. 353–361, 1996.
[32] Y. Polyanskiy and Y. Wu, “Lecture notes on information theory,”
Lecture Notes for 6.441 (MIT), ECE563 (University of Illinois Urbana-
Champaign), and STAT 664 (Yale), 2012-2017.
[33] D. P. Palomar and Y. Jiang, MIMO transceiver design via majorization
theory.
Now Publishers Inc, 2007.
[34] K. Shen and W. Yu, “Fractional programming for communication
systems—part I: Power control and beamforming,” IEEE Trans. Signal
Process., vol. 66, no. 10, pp. 2616–2630, 2018.
[35] T. M. Cover and J. A. Thomas, Elements of Information Theory.
John
Wiley & Sons, 2012.
[36] N. Shlezinger, M. Chen, Y. C. Eldar, H. V. Poor, and S. Cui, “UVeQFed:
Universal vector quantization for federated learning,” IEEE Trans. Signal
Process., vol. 69, pp. 500–514, 2021.
[37] N. Perraudin, J. Paratte, D. Shuman, L. Martin, V. Kalofolias, P. Van-
dergheynst, and D. K. Hammond, “GSPBOX: A toolbox for signal
processing on graphs,” arXiv preprint arXiv:1408.5781, 2014.
[38] W. Hu, G. Cheung, A. Ortega, and O. C. Au, “Multiresolution graph
fourier transform for compression of piecewise smooth images,” IEEE
Transactions on Image Processing, vol. 24, no. 1, pp. 419–433, 2015.
[39] A. W. Marshall, I. Olkin, and B. C. Arnold, Inequalities: theory of
majorization and its applications.
Springer, 1979, vol. 143.
13
