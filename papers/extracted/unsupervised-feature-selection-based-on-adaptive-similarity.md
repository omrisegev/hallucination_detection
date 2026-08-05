---
source_pdf: papers/Unsupervised Feature Selection based on Adaptive Similarity Learning and Subspace Clustering.pdf
slug: unsupervised-feature-selection-based-on-adaptive-similarity
pages: 22
extracted_on: 2026-08-05
---

# Unsupervised Feature Selection based on Adaptive Similarity Learning and Subspace Clustering

## Page 1

Unsupervised Feature Selection based on Adaptive
Similarity Learning and Subspace Clustering
Mohsen Ghassemi Parsaa, Hadi Zarea,∗, Mehdi Ghateeb
aFaculty of New Sciences and Technologies, University of Tehran, Iran
bDepartment of Computer Science, Amirkabir University of Technology, Iran
Abstract
Feature selection methods have an important role on the readability of data
and the reduction of complexity of learning algorithms. In recent years, a
variety of eﬀorts are investigated on feature selection problems based on un-
supervised viewpoint due to the laborious labeling task on large datasets.
In this paper, we propose a novel approach on unsupervised feature selec-
tion initiated from the subspace clustering to preserve the similarities by
representation learning of low dimensional subspaces among the samples. A
self-expressive model is employed to implicitly learn the cluster similarities
in an adaptive manner. The proposed method not only maintains the sample
similarities through subspace clustering, but it also captures the discrimina-
tive information based on a regularized regression model. In line with the
convergence analysis of the proposed method, the experimental results on
benchmark datasets demonstrate the eﬀectiveness of our approach as com-
pared with the state of the art methods.
Keywords:
Unsupervised feature selection, Graph learning, Subspace
clustering, Sparse learning, Representation learning
1. Introduction
One of the most common approaches for dealing with high dimensional
data is to select the appropriate features which is known as feature selec-
tion (FS) problem in machine learning community [1, 2]. FS techniques are
widely applied in many domains including text mining [3], bioinformatics [4],
∗Corresponding author
arXiv:1912.05458v1  [cs.LG]  10 Dec 2019

## Page 2

social media [5], and ensemble learning [6]. On the one hand, FS approach
provides a sparse representation for data with massive number of features
to alleviate the curse of dimensionality eﬀect on the learning performance
[7]. On the other hand, the computational burden of facing with massive
data could be decreased signiﬁcantly through FS approach as compared with
other techniques like feature extraction approaches [8].
FS techniques can be categorized into wrapper, ﬁlter and embedded ap-
proaches by considering the evaluation criteria. In wrapper methods [9, 10,
11], the evaluation process depends on the learning algorithms, in contrast
to the ﬁlter methods [12, 13, 14] that only use data for evaluating without
any learning phase. The embedded methods [15, 16], embed the process of
selecting features in a learning algorithm.
On learning taxonomy, FS problem can be stated as “Supervised” and
“Unsupervised”. Supervised FSs are primarily constructed from the depen-
dency among the features and the label information, including information
theoretic approaches [17], statistical tests [18, 19], sparse learning methods
[20, 21], and structure learning [22]. On the other hand, appropriate cri-
terion on Unsupervised FS (UFS) problems is more challenging due to the
ill-deﬁned nature of the problem. There have been a variety of works on
UFS [23] including metaheuristic approach [24], graph clustering [25], feature-
level reconstruction [26], discriminative approach [27], and spectral clustering
[28, 29]. One of the important aims in UFS approaches is to preserve the
geometric structure in the selected features [13]. To this end, the similarity
preserving methods construct a graph similarities to maintain the geometric
structure in the reduced space [30].
The graph similarity computation is often performed independently from
the feature selection, which may lead to a suboptimal solution. Adaptive
structure learning methods [31] explicitly learn a similarity matrix which
expand the search space of solutions. There are some UFS by considering
the subspace learning to exploit the hidden multidimensional substructures
of data [32]. The main idea of the subspace learning is constructed from
the representation of high dimensional data points based on the union of
the subspaces [33]. Subspace clustering refers to the process of clustering
and detecting the low dimensional structure of the clusters at the same time
[34].
The earlier subspace learning methods [35, 36] have considered the
self-expressiveness of the features aligned with the graph similarity stage.
In this paper, we propose a novel UFS, “Subspace Clustering unsuper-
vised Feature Selection” (SCFS), to exploit the discriminative information
2

## Page 3

Cluster 
analysis
Subspace 
learning
Data matrix
Selected 
features
Sparse 
learning
Figure 1: The outline of the proposed method.
concurrent with cluster analysis and adaptively maintain the similarity struc-
ture through an implicit similarity matrix computation. The proposed ap-
proach is constituted by a self-expressive model to include both of learning
latent homogeneous structures and similarity matrix computation in a uni-
ﬁed objective function aligned with an ℓ2,1-norm to address a regularized
regression model. In our approach, subspace learning is utilized to maintain
the cluster similarities in the selected features and sparse learning is applied
to learn a regularized regression model to measure the correlation between
the features and the learned clustering information. The more a feature is
related to the clusters, the more it is likely to be selected. The outline of the
proposed method is presented in Fig. 1.
The main contributions of this paper are as follows,
• We propose a novel UFS method by applying subspace learning, cluster
analysis and sparse learning to consider the sample similarities and
discriminative information in the selected features.
• We introduce a self-expressive model to adaptively and implicitly learn
the cluster similarities.
• We use a regularized regression approach to compute the sparse corre-
lation among the features and clusters.
• We introduce an optimization algorithm to address the proposed ob-
jective function.
3

## Page 4

This paper is organized as follows.
The related works are introduced in
Section 2. The proposed method and the corresponding optimization algo-
rithm are presented in Section 3. Convergence analysis and computational
complexity are discussed in Section 4. Experimental setting and results are
reported in Section 5. Finally, the conclusion of the paper is provided in
Section 6.
2. Related Works
Unsupervised feature selection methods generally select features based on
the intrinsic structural characteristics of data. These methods can be divided
into three main categories, similarity preserving [13], data reconstruction [37],
and sparse learning approaches [28].
The main focus of similarity preserving methods including Laplacian
Score [13] and SPEC [14], is to maintain the local similarities. Reconstruction
based methods employ a feature-level self-expressive model including con-
vex principal feature selection, CPFS [37], embedded reconstruction based,
REFS [26] and structure preserving, SPUFS [38]. Sparse unsupervised FS
approaches are initiated from the ideas of the sparse machine learning [39].
The core idea is to embed FS in a regularized learning model. There are sev-
eral well-known approaches in this category including preserving the multi-
cluster structure of data, MCFS [40], local discriminative approach using the
scatter matrix, UDFS [27], joint embedding learning and sparse regression,
JELSR [15], nonnegative discriminative feature selection based on spectral
clustering, NDFS [28], global similarity preserving feature selection, SPFS
[41], global and local similarity preserving feature selection, GLSPFS [30],
and unsupervised feature selection with adaptive structure learning, FSASL
[31].
There are some UFS methods by considering the subspace learning idea.
Feature-level reconstruction based approach was proposed in, MFFS [32] by
exploiting the matrix factorization. In [35], a graph regularized approach
was introduced to maintain the local similarities. A sparse discriminative
learning approach was devised in [36] to select discriminative features based
on the local structure of the samples. While, UFS methods with the aid of
subspace learning, MFFS [32], SGFS [35], LDSSL [36] mainly reconstruct
the data matrix in feature-level, the sample-level characteristics such as the
cluster structures are not thoroughly incorporated in them.
4

## Page 5

Table 1: A comparison of the related unsupervised feature selection methods.
Algorithm
Self-
expression
Similarity preserving
Adaptive
graph matrix
Joint
learning
Cluster
analysis
Regression
Regularization
LS [13]
×
Explicit
×
✓
×
×
×
CPFS [37]
Feature
×
×
×
×
×
✓
MCFS [40]
×
Explicit
×
×
×
✓
✓
UDFS [27]
×
Explicit
×
✓
×
✓
✓
NDFS [28]
×
Explicit
×
✓
✓
✓
✓
GLSPFS [30]
×
Explicit
×
✓
×
✓
✓
MFFS [32]
Feature
×
×
×
×
×
×
FSASL [31]
×
Explicit
✓
✓
×
✓
✓
REFS [26]
Feature
Explicit
×
✓
×
×
×
SPUFS [38]
Feature
Explicit
×
✓
×
×
✓
LDSSL [36]
Feature
Explicit
×
✓
×
×
✓
SCFS
Sample
Implicit
✓
✓
✓
✓
✓
The important properties of the well-known UFS methods are summa-
rized in Table 1.
Theses methods are compared based on multiple prop-
erties including, Self-expression, Similarity preserving, Adaptive graph ma-
trix, Joint learning, Cluster analysis, Regression and Regularization. Self-
expressive property points out the reconstruction on samples or features.
Similarity preserving is related to maintain sample similarities and comput-
ing explicitly or implicitly of the similarity matrix. Adaptive graph matrix
indicates the property of learning the similarity matrix concurrent with the
feature selection process. Joint learning refers to perform both of subspace
learning and feature selection in a uniﬁed framework. Cluster analysis in-
dicates that a method employs any clustering algorithm to select relevant
features. Regression refers to exploit a regression model to discover discrim-
inative features. Finally, Regularization indicates the consideration of regu-
larization factors in the method to result a sparse solution. Most of the UFS
techniques are constructed from one or more of these properties, but in this
work, a uniﬁed framework is proposed to consider all of the characteristics
to provide a more robust UFS.
3. The Proposed Method
3.1. Notations
Throughout this paper, matrices are denoted by bold uppercase and vec-
tors by bold lowercase characters. Let B be an arbitrary matrix , Bij is
its (i, j)-th element, and bi denotes the i-th row. The Frobenius norm, the
trace and the transpose operators on matrix B are denoted by ∥B∥F, tr(B),
and B⊤, respectively. The ℓ2-norm of a vector v is denoted as ∥v∥2 and the
5

## Page 6

ℓ2,1-norm is deﬁned as following,
∥B∥2,1 =
X
i
sX
j
B2
ij.
X ∈Rn×p denotes the data matrix, where n and p are the number of
samples and features. G ∈Rn×c represents the clustering matrix, where c is
the number of clusters.
3.2. The Proposed Method
At ﬁrst, the similarity matrix is implicitly computed by subspace learning.
The proposed self-expressive similarity representation is given in Eq. (1),
min
G
∥X −GG⊤X∥
2
F
s.t.
G ≥0, GG⊤1 = 1,
(1)
where 1 is an n×n matrix of ones and the constraint GG⊤1 = 1 is imposed to
normalize the similarity matrix. The symmetric nonnegative matrix GG⊤
is learned such that the samples within common subspaces tend to attain
large values in GG⊤. In lines with a low dimensional representation of G
by assuming c < {n, p}, G can also be interpreted as a clustering matrix.
Moreover, GG⊤, represents the pairwise sample similarities in terms of the
clustering values.
The next stage is to construct a sparse transformation W on the data
matrix X by employing the clustering matrix G, joined with a regularization
term,
min
W
∥XW −G∥2
F + β∥W∥2,1,
(2)
where W ∈Rp×c is a linear and low dimensional transformation matrix, and
β is a regularization parameter. The objective function in Eq. (2) represents
the linear transformation model to measure the association between features
and clusters. The ℓ2,1 norm induces sparsity on the rows of the transformation
matrix, wi’s. When wi’s are closer to zero, their correspondence features are
less relevant and more likely to be eliminated from the ﬁnal candidate set of
the discriminative features.
By integrating Eq. (1) and (2) in a joint objective function, our ﬁnal
model is obtained as follows,
min
W,G
∥X −GG⊤X∥
2
F + α∥XW −G∥2
F + β∥W∥2,1
s.t.
G ≥0, GG⊤1 = 1,
(3)
6

## Page 7

Similarity Matrix: GGT
Regression Matrix: W
F6 (Selected)
F9 (Selected)
F3 (Selected)
Data Matrix: X
Samples
Features
Subspace 
learning
Cluster 
analysis
Sparse 
learning
Sparse 
learning
Cluster 
analysis
Feature 
selection
Clustering Matrix: G
Selected features
F6 F3 F9
C1 C2 C3
S1
S2
S3
S4
S5
S6
S7
F1 F2 F3 F4 F5 F6 F7 F8 F9 F10
Figure 2: The description of the proposed method.
where α is a tuning parameter. By solving the objective function in Eq. (3),
W and G are iteratively updated in advance of achieving the optimal result.
We illustrate the steps of SCFS in Fig. 2. X is a nonnegative artiﬁcial
data matrix with seven samples and ten features. The bright entries of X
are close to zero and the dark ones are far from zero. Initially, subspace
learning stage provides the cluster similarities G which is used to construct
the similarity matrix GG⊤among samples. Then, a sparse learning method
is applied for learning the regularized coeﬃcients W through a regression
model to measure the importance of features. The W and G are optimized
in an iterative process. Finally, the most important features are selected
based on W. In this example, F6, F3, and F9 are selected according to their
roles’ in the learned hidden subspaces.
3.3. Optimization
The primary objective function in Eq. (3) can be considered as,
min
W,G≥0,GG⊤1=1
f(W, G) = ∥X −GG⊤X∥
2
F + α∥XW −G∥2
F + β∥W∥2,1.
(4)
7

## Page 8

A gradient based procedure is utilized to solve this optimization problem by
considering the main elements W and G. It begins by ﬁxing one element and
ﬁnding the optimum value for the other ones which is described in below.
Initially, G is ﬁxed to yield the following objective function,
min
W
f(W) = α∥XW −G∥2
F + β∥W∥2,1.
(5)
Taking the derivative to calculate the ∇f(W) and setting it to zero,
W =
 αX⊤X + βD
−1αX⊤G,
(6)
where D is a diagonal matrix with,
Dii =
1
2∥wi∥2 + ϵ,
(7)
where ϵ is a very small positive number to prevent the division by zero.
Then, G is updated through the objective function in Eq. (8) by ﬁxing W,
min
G≥0,GG⊤1=1
f(G) = ∥X −GG⊤X∥
2
F + α∥XW −G∥2
F.
(8)
Eq. (8) is rewritten to relax the constraints as,
f(G) = ∥X −GG⊤X∥
2
F + α∥XW −G∥2
F + γ∥GG⊤1 −1∥
2
F + tr
 ΦG⊤
,
(9)
where γ > 0 is a parameter to control the normalizing constraint and prac-
tically should be a large number. Φ is the Lagrange multiplier for G ≥0
constraint. Setting the derivative of f(G) with respect to G to 0,
2MG⊤G + 2GG⊤M + 2αG −4M −2αXW + Φ = 0,
(10)
where M =
 XX⊤+ nγ1

G.
By applying the KKT condition [42], the
following updating rule is obtained,
Gij = Gij
[2M + αXW]ij
[MG⊤G + GG⊤M + αG]ij
,
(11)
Therefore, by initializing the G and D, in each iteration of the proposed
formulation, ﬁrst W is updated by Eq. (6), and then G and D is updated
by Eq. (11) and (7). Algorithm 1 describes the optimization process of the
proposed method.
8

## Page 9

Algorithm 1 SCFS algorithm.
Input: Data matrix X ∈Rn×p and parameters α and β.
1: t = 0.
2: Initialize G0 ∈Rn×c.
3: Initialize D0 as an identity matrix.
4: repeat
5:
Wt+1 =
 αX⊤X + βDt
−1αX⊤Gt.
6:
Mt =
 XX⊤+ nγ1

Gt.
7:
(Gt+1)ij = (Gt)ij
[2Mt + αXWt+1]ij
[MtG⊤
t Gt + GtG⊤
t Mt + αGt]ij
.
8:
Update the diagonal matrix D as (Dt+1)ii =
1
2∥(wt+1)i∥2+ϵ.
9:
t = t + 1.
10: until Convergence of the objective function in Eq. (3).
Output: Sort features by descending order of ∥wi∥2.
4. The analysis of the proposed algorithm
This section presents the convergence behavior and computational com-
plexity of SCFS.
4.1. Convergence Analysis
Our aim is to show the non-increasing behavior of the primary objective
function in Eq. (3). First, a lemma is given [21].
Lemma 1. For any nonzero vectors u, v ∈Rp, the following holds,
∥u∥2 −∥u∥2
2
2∥v∥2
≤∥v∥2 −∥v∥2
2
2∥v∥2
.
(12)
Theorem 1. The objective function in Eq. (3) is non-increasing in each
iteration by employing the updating rules in Algorithm 1.
Proof. First, the objective function can be written as,
f(W, G) =
∥X −GG⊤X∥
2
F + α∥XW −G∥2
F + β∥W∥2,1
+γ∥GG⊤1 −1∥
2
F.
(13)
9

## Page 10

By ﬁxing Gt, we should justify the following inequality,
f(Wt+1, Gt) ≤f(Wt, Gt).
(14)
Based on Eq. (5), inequality (14) can be written as,
∥XWt+1 −Gt∥2
F + β Pp
i=1(∥(wt+1)i∥2
2
2∥(wt)i∥2 )
≤∥XWt −Gt∥2
F + β Pp
i=1( ∥(wt)i∥2
2
2∥(wt)i∥2).
(15)
The inequality (15) is followed as,
∥XWt+1 −Gt∥2
F + β∥Wt+1∥2,1 −β
p
X
i=1
(∥(wt+1)i∥2 −∥(wt+1)i∥2
2
2∥(wt)i∥2
)
≤∥XWt −Gt∥2
F + β∥Wt∥2,1 −β
p
X
i=1
(∥(wt)i∥2 −∥(wt)i∥2
2
2∥(wt)i∥2
).
(16)
According to Lemma 1,
∥XWt+1 −Gt∥2
F + β∥Wt+1∥2,1 ≤∥XWt −Gt∥2
F + β∥Wt∥2,1.
(17)
Taking ﬁxed Wt+1, based on a similar approach in [43], it follows,
f(Wt+1, Gt+1) ≤f(Wt+1, Gt).
(18)
Hence,
f(Wt+1, Gt+1) ≤f(Wt+1, Gt) ≤f(Wt, Gt).
(19)
Therefore, Algorithm 1 will monotonically decrease the objective function in
Eq. (3) based on the relations (17) and (18).
4.2. Computational complexity
The main steps of Algorithm 1 contains the updating W and G on each
iteration. The update of W and G take O(p3 + np2 + npc) and O(n2p +
n2c + npc) time complexity. Hence, the time complexity of the proposed
algorithm is max{O(p3), O(np2), O(n2p), O(n2c), O(npc)}. In most applied
scenarios c ≪p, that implies the time complexity of the proposed algorithm
could be reduced to, max{O(p3), O(n2p)}.
10

## Page 11

Table 2: The main properties of datasets in the experiments.
Dataset
n
p
c
Type
Domain
Lung
203
3312
5
Continuous
Biology
Lymphoma
96
4026
9
Discrete
Biology
Prostate-GE
102
5966
2
Continuous
Biology
ORL
400
1024
40
Discrete
Image
Isolet
1560
617
26
Continuous
Voice
BASEHOCK
1993
4862
2
Discrete
Text
5. Experiments
In this section, the proposed method is evaluated using benchmark datasets
by standard evaluation measures. A bunch of state-of-the-art FS methods are
compared with SCFS where the results and experimental setting are reported
in the following.
5.1. Datasets
A variety of datasets are applied in diﬀerent domains including biologi-
cal (Lung, Lymphoma, Prostate-GE), image (ORL), voice (Isolet), and text
(BASEHOCK) data.
All of the datasets are available on repository [2].
Table 2 reports the main characteristics of datasets.
5.2. Evaluation measures
The performance is evaluated in terms of clustering by two widely used
and standard measures, Accuracy (Acc) and Normalized Mutual Information
(NMI). By taking y as the ground truth label information, and z as the
predicted ones’, Acc is deﬁned as,
Acc(y, z) = 1
n
n
X
i=1
δ(yi, map(zi)),
where δ(a, b) equals to 1 if a = b and 0, otherwise. The best permutation of
z to match y values is found by map(.) function based on the Kuhn-Munkres
approach [44]. The deﬁnition of NMI is given as,
NMI(y, z) =
I(y, z)
max(H(y), H(z)),
11

## Page 12

where H(.) represents the entropy and I(y, z) is the mutual information of
y and z deﬁned as,
I(y, z) =
X
y∈y
X
z∈z
p(y, z) log
 p(y, z)
p(y) p(z)

.
5.3. The experimental setting
The state-of-the-art UFS methods are applied such as LS[13], UDFS[27],
NDFS[28], SPUFS [38], LDSSL [36], and Baseline means to select all of the
original features.
We set k = 5 on k-nearest neighbor algorithm, and σ = 1 for the band-
width parameter in the Gaussian kernel for the methods based on explicit
construction of the graph matrix. The γ = 106 is taken on our method and
NDFS. The grid search strategy is employed to choose the appropriate weight
parameters α and β among the set of {10−4, 10−2, 1, 102, 104} candidates. We
limit data by selecting diﬀerent number of features in the range of {50, 100,
150, 200, 250, 300} and cluster each ones by k-means algorithm, and then
evaluate the clustering results by Acc and NMI measures. The mean and
standard deviation values of Acc and NMI are reported by repeating the
experiments for 20 times.
5.4. Experimental results
The performance of the feature selection algorithms are empirically eval-
uated in terms of Acc and NMI. The mean and standard deviation of the
clustering result are reported in the Table 3 and 4. The best and the second
best results are marked as bold and underline. By considering the Table
3 and 4, we have the following conclusions,
• The proposed approach outperform the Baseline method which is showed
the eﬃcacy of SCFS to select the more relevant features rather than
the irrelevant and redundant ones’.
• Clustering based methods such as NDFS and SCFS commonly attain
better results in an unsupervised manner.
• The proposed method, SCFS, achieves the best performance on Acc on
the whole datasets, and also the best on NMI on the most cases.
12

## Page 13

Table 3: Clustering results (Acc% ± std) of unsupervised feature selection methods
on standard datasets. Bold and underlined numbers are the best and the second
best.
Dataset
Lung
Lymphoma
Prostate-GE
ORL
Isolet
BASEHOCK
Baseline
71.67 ± 6.86
58.75 ± 5.19
58.82 ± 0.00
59.14 ± 2.11
63.19 ± 2.19
50.08 ± 0.00
LS
61.46 ± 2.61
50.12 ± 2.26
60.82 ± 1.71
49.94 ± 3.62
53.31 ± 4.49
50.63 ± 0.23
UDFS
56.65 ± 4.93
59.60 ± 2.50
61.36 ± 1.14
49.30 ± 3.90
57.97 ± 6.24
51.57 ± 0.56
NDFS
83.71 ± 0.64
63.81 ± 0.33
60.07 ± 0.77
58.81 ± 1.19
67.72 ± 1.51
50.18 ± 0.14
SPUFS
68.33 ± 1.20
53.57 ± 3.42
61.09 ± 1.45
48.63 ± 2.68
63.65 ± 13.10
50.41 ± 0.40
LDSSL
64.59 ± 5.14
58.11 ± 1.67
60.27 ± 0.74
57.78 ± 1.49
65.97 ± 3.64
50.49 ± 0.11
SCFS
86.70 ± 1.38
64.87 ± 1.59
61.70 ± 0.73
59.19 ± 0.83
69.17 ± 1.03
51.95 ± 0.44
Table 4: Clustering results (NMI% ± std) of unsupervised feature selection meth-
ods on standard datasets.
Bold and underlined numbers are the best and the
second best.
Dataset
Lung
Lymphoma
Prostate-GE
ORL
Isolet
BASEHOCK
Baseline
62.90 ± 2.76
68.95 ± 3.63
2.55 ± 0.00
77.90 ± 0.86
77.61 ± 1.12
0.63 ± 0.00
LS
50.16 ± 5.95
55.88 ± 2.52
4.46 ± 1.65
70.90 ± 2.67
70.41 ± 4.52
2.54 ± 0.82
UDFS
45.73 ± 4.02
69.46 ± 3.36
5.06 ± 1.12
70.75 ± 2.91
71.03 ± 6.07
1.02 ± 0.77
NDFS
67.68 ± 0.85
73.70 ± 0.71
5.42 ± 0.48
77.61 ± 0.72
79.40 ± 1.72
1.20 ± 0.81
SPUFS
60.28 ± 1.81
63.24 ± 2.45
5.17 ± 0.45
70.25 ± 2.24
72.81 ± 11.14
1.86 ± 1.25
LDSSL
52.31 ± 5.19
64.64 ± 1.66
5.66 ± 0.13
76.41 ± 1.26
77.26 ± 3.37
1.67 ± 0.39
SCFS
70.17 ± 0.90
73.73 ± 0.75
5.85 ± 0.46
77.71 ± 0.44
79.43 ± 1.62
3.73 ± 0.50
• Moreover, the proposed method outperforms the earlier subspace learn-
ing based approach, LDSSL, due to employing the sample-level self-
expression, and adaptive learning of the cluster similarities.
Furthermore, we demonstrate the performance of the proposed method for
two extreme scenarios, the ﬁrst by considering the number of selected features
as 50, and the second as 300. Fig. 3 and Fig. 4 represent the obtained results
according to these scenarios. On the one hand, SCFS performs satisfactory
on the ﬁrst scenario to deal with the small number of selected features. On
the other hand, the results indicate that the proposed approach attains better
performance than the other well-known methods on almost all datasets on
the second scenario.
13

## Page 14

50
300
40
60
80
Number of selected features
Acc (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(a) Lung
50
300
45
50
55
60
65
Number of selected features
Acc (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(b) Lymphoma
50
300
55
60
Number of selected features
Acc (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(c) Prostate-GE
50
300
40
45
50
55
60
Number of selected features
Acc (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(d) ORL
50
300
35
45
55
65
75
Number of selected features
Acc (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(e) Isolet
50
300
50
51
52
Number of selected features
Acc (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(f) BASEHOCK
Figure 3: The obtained results in terms of Acc with 50 and 300 numbers of selected
features.
50
300
30
40
50
60
70
Number of selected features
NMI (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(a) Lung
50
300
50
55
60
65
70
75
Number of selected features
NMI (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(b) Lymphoma
50
300
1
2
3
4
5
6
7
Number of selected features
NMI (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(c) Prostate-GE
50
300
65
70
75
Number of selected features
NMI (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(d) ORL
50
300
50
60
70
80
Number of selected features
NMI (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(e) Isolet
50
300
0
1
2
3
4
5
Number of selected features
NMI (%)
LS
UDFS
NDFS
SPUFS
LDSSL
SCFS
(f) BASEHOCK
Figure 4: The obtained results in terms of NMI with 50 and 300 numbers of
selected features.
14

## Page 15

5.5. Parameter sensitivity and convergence study
First, the sensitivity of parameters α and β in our model are investigated.
The experimental results on Acc and NMI criteria for all of datasets are
presented on Fig. 5. For all candidate of α and β parameters, the logarithms
base 10 is taken. As shown in Fig. 5, there is a relative sensitivity to the
parameters, which is still an open problem.
Next, we experimentally study the convergence behavior of the proposed
algorithm. Fig. 6 presents the speed of the convergence according to the ob-
jective values with respect to the number of iterations on diﬀerent datasets.
The stopping criteria is set as obj(t)−obj(t−1)
obj(t)
< 10−5, where obj(t) is the objec-
tive function value of Eq. (3) in the t-th iteration. As shown in the Fig. 6,
the proposed algorithm monotonically decreases the objective function in a
few iteration.
6. Conclusion
In this paper, we proposed a novel unsupervised feature selection frame-
work initiated from the subspace learning and regularized regression to main-
tain sample similarities and take discriminative information into account in
the selected features. The proposed method, SCFS, was designed to implic-
itly learn the cluster similarities in an adaptive manner. Furthermore, a uni-
ﬁed objective function was constituted from the main underlying character-
istics of the proposed method. The optimization algorithm was proposed to
obtain the solutions in an eﬃcient way. In line with the computational com-
plexity of the proposed algorithm, its convergence was investigated through
an empirical study on real datasets. Extensive experiments on variaty of
datasets was performed to show the eﬀectiveness of the proposed method.
References
References
[1] I. Guyon, A. Elisseeﬀ, An Introduction to Variable and Feature Selec-
tion, Journal of Machine Learning Research 3 (2003) 1157–1182.
[2] J. Li, K. Cheng, S. Wang, F. Morstatter, R. P. Trevino, J. Tang, H. Liu,
Feature Selection: A Data Perspective, http://featureselection.asu.edu/,
ACM Computing Surveys 50 (6) (2017) 94:1–94:45.
15

## Page 16

[3] T. Liu, S. Liu, Z. Chen, W.-Y. Ma, An Evaluation on Feature Selec-
tion for Text Clustering, in: Proceedings of the Twentieth International
Conference on International Conference on Machine Learning, ICML’03,
AAAI Press, 2003, pp. 488–495.
[4] D. C. H.q., Unsupervised feature selection via two-way ordering in gene
expression analysis, Bioinformatics 19 (10) (2003) 1259.
[5] J. Tang, H. Liu, An Unsupervised Feature Selection Framework for So-
cial Media Data, IEEE Transactions on Knowledge and Data Engineer-
ing 26 (12) (2014) 2914–2927.
[6] S. Abpeykar, M. Ghatee, H. Zare, Ensemble Decision Forest of RBF Net-
works via Hybrid Feature Clustering Approach for High-Dimensional
Data Classiﬁcation, Computational Statistics & Data Analysis 131
(2019) 12–36.
[7] C. M. Bishop, Pattern Recognition and Machine Learning (Information
Science and Statistics), Springer-Verlag, Berlin, Heidelberg, 2006.
[8] I. Guyon (Ed.), Feature extraction: foundations and applications, no.
207 in Studies in fuzziness and soft computing, Springer, Berlin, 2006.
[9] J. G. Dy, C. E. Brodley, Feature Selection for Unsupervised Learning,
Journal of Machine Learning Research 5 (2004) 845–889.
[10] R. Kohavi, G. H. John, Wrappers for Feature Subset Selection, Artiﬁcial
Intelligence 97 (1-2) (1997) 273–324.
[11] V. Roth, T. Lange, Feature selection in clustering problems, MIT Press,
2003, pp. 473–480.
[12] P. Mitra, C. A. Murthy, S. K. Pal, Unsupervised Feature Selection Using
Feature Similarity, IEEE Transactions on Pattern Analysis and Machine
Intelligence 24 (3) (2002) 301–312.
[13] X. He, D. Cai, P. Niyogi, Laplacian Score for Feature Selection, in:
Proceedings of the 18th International Conference on Neural Information
Processing Systems, NIPS’05, MIT Press, Cambridge, MA, USA, 2005,
pp. 507–514.
16

## Page 17

[14] Z. Zhao, H. Liu, Spectral Feature Selection for Supervised and Unsu-
pervised Learning, in: Proceedings of the 24th International Conference
on Machine Learning, ICML ’07, ACM, New York, NY, USA, 2007, pp.
1151–1157.
[15] C. Hou, F. Nie, X. Li, D. Yi, Y. Wu, Joint Embedding Learning and
Sparse Regression: A Framework for Unsupervised Feature Selection,
IEEE Transactions on Cybernetics 44 (6) (2014) 793–804.
[16] X. Han, P. Liu, L. Wang, D. Li, Unsupervised feature selection via graph
matrix learning and the low-dimensional space learning for classiﬁcation,
Engineering Applications of Artiﬁcial Intelligence 87 (2020) 103283.
[17] H. Peng, F. Long, C. Ding, Feature Selection Based on Mutual In-
formation:
Criteria of Max-Dependency, Max-Relevance, and Min-
Redundancy, IEEE Transactions on Pattern Analysis and Machine In-
telligence 27 (8) (2005) 1226–1238.
[18] M. A. Hall, L. A. Smith, Feature Selection for Machine Learning: Com-
paring a Correlation-Based Filter Approach to the Wrapper, in: Pro-
ceedings of the Twelfth International Florida Artiﬁcial Intelligence Re-
search Society Conference, AAAI Press, 1999, pp. 235–239.
[19] Huan Liu, R. Setiono, Chi2: feature selection and discretization of nu-
meric attributes, in: Proceedings of 7th IEEE International Conference
on Tools with Artiﬁcial Intelligence, 1995, pp. 388–391.
[20] J. Liu, S. Ji, J. Ye, Multi-task Feature Learning via Eﬃcient L2, 1-
norm Minimization, in: Proceedings of the Twenty-Fifth Conference on
Uncertainty in Artiﬁcial Intelligence, UAI ’09, AUAI Press, Arlington,
Virginia, United States, 2009, pp. 339–348.
[21] F. Nie, H. Huang, X. Cai, C. Ding, Eﬃcient and Robust Feature Se-
lection via Joint L2,1-norms Minimization, in: Proceedings of the 23rd
International Conference on Neural Information Processing Systems -
Volume 2, NIPS’10, Curran Associates Inc., USA, 2010, pp. 1813–1821.
[22] H. Zare, M. Niazi, Relevant based structure learning for feature selec-
tion, Engineering Applications of Artiﬁcial Intelligence 55 (2016) 93–102.
17

## Page 18

[23] S. Solorio-Fernndez, J. A. Carrasco-Ochoa, J. F. Martnez-Trinidad, A
review of unsupervised feature selection methods, Artiﬁcial Intelligence
Review (2019) 1–42.
[24] S. Tabakhi, P. Moradi, F. Akhlaghian, An unsupervised feature selection
algorithm based on ant colony optimization, Engineering Applications
of Artiﬁcial Intelligence 32 (2014) 112–123.
[25] P. Moradi, M. Rostami, A graph theoretic approach for unsupervised
feature selection, Engineering Applications of Artiﬁcial Intelligence 44
(2015) 33–45.
[26] J. Li, J. Tang, H. Liu, Reconstruction-based Unsupervised Feature Selec-
tion: An Embedded Approach, in: Proceedings of the 26th International
Joint Conference on Artiﬁcial Intelligence, IJCAI’17, AAAI Press, 2017,
pp. 2159–2165.
[27] Y. Yang, H. T. Shen, Z. Ma, Z. Huang, X. Zhou, L2,1-norm Regularized
Discriminative Feature Selection for Unsupervised Learning, in: Pro-
ceedings of the Twenty-Second International Joint Conference on Arti-
ﬁcial Intelligence - Volume Volume Two, IJCAI’11, AAAI Press, 2011,
pp. 1589–1594.
[28] Z. Li, Y. Yang, J. Liu, X. Zhou, H. Lu, Unsupervised Feature Selection
Using Nonnegative Spectral Analysis, in: Proceedings of the Twenty-
Sixth AAAI Conference on Artiﬁcial Intelligence, AAAI’12, AAAI Press,
2012, pp. 1026–1032.
[29] Z. Li, J. Liu, Y. Yang, X. Zhou, H. Lu, Clustering-Guided Sparse Struc-
tural Learning for Unsupervised Feature Selection, IEEE Transactions
on Knowledge and Data Engineering 26 (9) (2014) 2138–2150.
[30] X. Liu, L. Wang, J. Zhang, J. Yin, H. Liu, Global and Local Struc-
ture Preservation for Feature Selection, IEEE Transactions on Neural
Networks and Learning Systems 25 (6) (2014) 1083–1095.
[31] L. Du, Y.-D. Shen, Unsupervised Feature Selection with Adaptive Struc-
ture Learning, in: Proceedings of the 21th ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining, KDD ’15, ACM,
New York, NY, USA, 2015, pp. 209–218.
18

## Page 19

[32] S. Wang, W. Pedrycz, Q. Zhu, W. Zhu, Subspace Learning for Unsuper-
vised Feature Selection via Matrix Factorization, Pattern Recognition
48 (1) (2015) 10–19.
[33] M. Soltanolkotabi, E. Elhamifar, E. J. Cands, Robust subspace cluster-
ing, The Annals of Statistics 42 (2) (2014) 669–699.
[34] R. Vidal, Subspace Clustering, IEEE Signal Processing Magazine 28 (2)
(2011) 52–68.
[35] R. Shang, W. Wang, R. Stolkin, L. Jiao, Subspace Learning-based Graph
Regularized Feature Selection, Knowledge-Based Systems 112 (C)
(2016) 152–165.
[36] R. Shang, Y. Meng, W. Wang, F. Shang, L. Jiao, Local discriminative
based sparse subspace learning for feature selection, Pattern Recognition
92 (2019) 219–230.
[37] M. Masaeli, Y. Yan, Y. Cui, G. Fung, J. Dy, Convex Principal Feature
Selection, in: Proceedings of the 2010 SIAM International Conference
on Data Mining, Proceedings, Society for Industrial and Applied Math-
ematics, 2010, pp. 619–628.
[38] Q. Lu, X. Li, Y. Dong, Structure Preserving Unsupervised Feature Se-
lection, Neurocomputing 301 (C) (2018) 36–45.
[39] F. Bach, R. Jenatton, J. Mairal, G. Obozinski, Optimization with
Sparsity-Inducing Penalties, Foundations and Trends in Machine Learn-
ing 4 (1) (2012) 1–106.
[40] D. Cai, C. Zhang, X. He, Unsupervised Feature Selection for Multi-
cluster Data, in: Proceedings of the 16th ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining, KDD ’10, ACM,
New York, NY, USA, 2010, pp. 333–342.
[41] Z. Zhao, L. Wang, H. Liu, J. Ye, On Similarity Preserving Feature Se-
lection, IEEE Transactions on Knowledge and Data Engineering 25 (3)
(2013) 619–632.
[42] H. W. Kuhn, A. W. Tucker, Nonlinear Programming, in: G. Giorgi,
T. H. Kjeldsen (Eds.), Traces and Emergence of Nonlinear Program-
ming, Springer Basel, Basel, 2014, pp. 247–258.
19

## Page 20

[43] D. D. Lee, H. S. Seung, Algorithms for Non-negative Matrix Factoriza-
tion, in: Proceedings of the 13th International Conference on Neural
Information Processing Systems, NIPS’00, MIT Press, Cambridge, MA,
USA, 2000, pp. 535–541.
[44] L. Lovasz, Matching Theory (North-Holland Mathematics Studies), El-
sevier Science Ltd., Oxford, UK, UK, 1986.
20

## Page 21

4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
80
100
60
20
40
0
Acc %
(a) Lung
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
40
20
80
60
0
Acc %
(b) Lymphoma
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
40
20
80
60
0
Acc %
(c) Prostate-GE
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
40
20
0
60
Acc %
(d) ORL
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
40
20
80
60
0
Acc %
(e) Isolet
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
40
20
0
60
Acc %
(f) BASEHOCK
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
40
20
80
60
0
NMI %
(g) Lung
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
40
20
80
60
0
NMI %
(h) Lymphoma
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
4
2
0
6
NMI %
(i) Prostate-GE
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
40
20
80
60
0
NMI %
(j) ORL
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
40
20
80
60
0
NMI %
(k) Isolet
4 
2 
0 
β
-2
-4
4 
2 
α
0 
-2
-4
2
2.5
1.5
0.5
1
0
NMI %
(l) BASEHOCK
Figure 5: Acc and NMI of SCFS with diﬀerent values of the parameters α and β.
21

## Page 22

0
5
10
15
20
25
30
5
5.5
6
·104
Number of itertions
Objective function value
(a) Lung
0
5
10
15
20
25
30
6.84
6.85
6.86
·105
Number of itertions
Objective function value
(b) Lymphoma
0
5
10
15
20
25
30
3
3.5
4
·104
Number of itertions
Objective function value
(c) Prostate-GE
0
5
10
15
20
25
30
0.5
1
·109
Number of itertions
Objective function value
(d) ORL
0
5
10
15
20
25
30
1.4
1.6
1.8
·105
Number of itertions
Objective function value
(e) Isolet
0
5
10
15
20
25
30
6.2
6.25
6.3
6.35
·105
Number of itertions
Objective function value
(f) BASEHOCK
Figure 6: Convergence curve of SCFS on diﬀerent datasets.
22
