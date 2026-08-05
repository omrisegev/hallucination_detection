---
source_pdf: papers/Multi-modal Differentiable Unsupervised Feature Selection.pdf
slug: multi-modal-differentiable-unsupervised-feature-selection
pages: 26
extracted_on: 2026-08-05
---

# Multi-modal Differentiable Unsupervised Feature Selection

## Page 1

Multi-modal Diﬀerentiable Unsupervised Feature
Selection
Junchen Yang 1
Oﬁr Lindenbaum 2
Yuval Kluger1
Ariel Jaﬀe3†
1Yale University, USA;
2Bar-Ilan University, Israel;
3Hebrew University of Jerusalem, Israel
†Corresponding author. E-mail: ariel.jaﬀe@mail.huji.ac.il
Abstract
Multi-modal high throughput biological data presents a great scientiﬁc oppor-
tunity and a signiﬁcant computational challenge. In multi-modal measurements,
every sample is observed simultaneously by two or more sets of sensors. In such
settings, many observed variables in both modalities are often nuisance and do
not carry information about the phenomenon of interest. Here, we propose a
multi-modal unsupervised feature selection framework: identifying informative
variables based on coupled high-dimensional measurements. Our method is
designed to identify features associated with two types of latent low-dimensional
structures: (i) shared structures that govern the observations in both modalities
and (ii) diﬀerential structures that appear in only one modality. To that end,
we propose two Laplacian-based scoring operators. We incorporate the scores
with diﬀerentiable gates that mask nuisance features and enhance the accuracy
of the structure captured by the graph Laplacian. The performance of the new
scheme is illustrated using synthetic and real datasets, including an extended
biological application to single-cell multi-omics.
1
arXiv:2303.09381v1  [cs.LG]  16 Mar 2023

## Page 2

1
Introduction
In an eﬀort to study biological systems, researchers are developing cutting-edge
techniques that measure up to tens of thousands of variables at single-cell resolution.
The complexity of such systems requires collecting multi-modal measurements to
understand the interplay between diﬀerent biological processes. Examples of such
multi-modal measurements include SHARE-seq [1], DBiT-seq [2], CITE-seq [3], etc.,
which have provided biological insights and advancements in applications such as
transcription factor characterization [4], cell type identiﬁcation in human hippocampus
[5], and immune cell proﬁling [6].
Multi-modal learning is a powerful tool widely used across multiple disciplines to
extract latent information from high-dimensional measurements [7, 8]. Humans use
complementary senses when attempting to “estimate” spoken words or sentences [9].
For example, lip movements can help us distinguish between two syllables that sound
similar. The same intuition has inspired statisticians and machine learning researchers
to develop learning techniques that exploit information captured simultaneously by
complementary measurement devices.
Due to their applicability in multiple domains, there has been a growing interest
in multi-modal approaches. Algorithms such as Contrastive Language–Image Pre-
training (CLIP) [10], and Audioclip [11] have pushed the performance boundaries of
machine learning for image, text, audio, analysis, and synthesis. The multi-modal data
fusion task dates back to [12], which proposed the celebrated Canonical Correlation
Analysis (CCA). CCA has many extensions [13, 14], and applications in diverse
scientiﬁc domains [15, 16]. Despite their tremendous success, classical or advanced
multi-modal schemes are often unsuitable for analyzing biological data. The large
number of nuisance variables, which often exceeds the number of measurements, often
causes correlation-based methods to overﬁt.
To attenuate the inﬂuence of nuisance or noisy features, several authors proposed
unsupervised feature selection (UFS) schemes [17].
UFS seeks small subsets of
informative variables in order to improve downstream analysis tasks, such as clustering
or manifold learning. Empirical results demonstrate that informative features are
often smooth with respect to some latent structure [18]. In practice, the smoothness
of features can be evaluated based on how slowly they vary with respect to a graph
[19]. Follow-up works exploited this idea to identify informative features [20, 21]. An
alternative paradigm for UFS seeks subsets of features that can be used to reconstruct
the entire data eﬀectively [22].
While most fusion methods focus on extracting information shared between modali-
ties, we propose a multi-modal UFS framework to identify features associated both with
structures that appear in both modalities, and structures that are modality-speciﬁc,
and appear in only one modality. To capture the shared structure, we construct a
2

## Page 3

symmetric shared graph Laplacian operator that enhances the shared geometry across
modalities. We further propose diﬀerential graph operators that capture smooth
structures that are not shared with the other modality. To perform multi-modal
feature selection, we incorporate diﬀerentiable gates [23, 24] with the shared and
modality-speciﬁc graph Laplacian scoring functions. This leads to a diﬀerentiable
UFS scheme that attenuates the inﬂuence of nuisance features during training and
computes a more accurate Laplacian matrix [25].
Our contributions are four folds: (i) Develop a shared and modality-speciﬁc Lapla-
cian scoring operators. (ii) Motivate our operators using a product of manifolds model.
(iii) develop and implement a diﬀerentiable framework for multi-modal UFS. (iv)
Evaluate the merits and limitations of our approach with synthetic and real data and
compare it to existing schemes.
2
Problem setting and preliminaries
We are given two data matrices X ∈Rn×d, Y ∈Rn×m whose rows contain n observa-
tions captured simultaneously in two modalities. The two sets of observations can
be, for example, two arrays of sensors, cameras with diﬀerent angles, etc. We are
interested in processing modalities with bijective correspondences, which implies that
there is a registration between the observations in both modalities.
Though the observations are high-dimensional, we assume that there are a small
number of parameters governing the physical processes that underlies the data. These
parameters can be continuous such as in a developmental process, or discrete - for
example, when the observations can be characterized by clustering. However, the
latent structure in both modalities may not be identical. For example, the two sets
of observations may be generated by sets of sensors with diﬀerent resolutions or
sensitivity. For illustration, consider the observations shown in Fig. 1 (left). Both
modalities follow a very similar tree structure. The bottom tree, however, has an
additional bifurcating point that does not appear in the upper tree (green points).
Thus, we assume the latent parameters can be partitioned into two subsets. The
ﬁrst component denoted θs, captures the structures shared by both modalities. The
second component, denoted θx for modality X, and θy for modality Y , captures the
modality-speciﬁc structures that only appear in one set of observations. For example,
the additional branch in the bottom tree (modality Y ) in Fig. 1 is governed by a
parameter in θy. Thus, the observations X and Y are nonlinear transformations of
θs, θx and θs, θy, respectively.
Many biological data modalities are high dimensional and contain noisy features,
which hinders the discovery of the underlying shared or modality-speciﬁc structures.
Here, our goal is to identify groups of features associated with the shared structures θs
3

## Page 4

(e.g., the groups of features that are smooth on the shared bifurcated tree in Fig. 1)
and groups of features associated with the modality-speciﬁc structures θx and θy (e.g.,
the features that are smooth with respect to the additional branch (θy) of modality
Y in Fig. 1). To achieve this goal, we compute two graphs that correspond to the
two modalities. We use a spectral method to uncover the shared and graph-speciﬁc
structures and apply a feature selection method to detect variables relevant to these
structures. To better understand our approach, we ﬁrst introduce some preliminaries
about graph representation in Sec. 2.1, and discuss related work on feature selection
in Sec. 2.2.
Figure 1: Overview of the goal: discovering features associated with shared and
modality speciﬁc latent structures
2.1
The graph Laplacian and Laplacian score
A common assumption when analyzing high-dimensional datasets is that their structure
lies on a low dimensional manifold in the high dimensional space [26, 27]. Methods
for manifold learning are often based on a graph that captures the aﬃnities between
data points. Let x(i), y(i) denote the i-th observation in the X and Y modalities and
let Kx, Ky be, respectively, their aﬃnity matrices whose elements are computed by
the following Gaussian kernel functions.
(Kx)i,j = exp

−∥x(i) −x(j)∥2
2σ2
x

,
(Ky)i,j = exp

−∥y(i) −y(j)∥2
2σ2
y

,
where σx, σy are user-deﬁned bandwidths that control the decay of each Gaussian
kernel. Intuitively, the aﬃnities decay exponentially with the distances between
samples, thus capturing the local neighborhood structure in the high-dimensional
space.
4

## Page 5

We compute the normalized Laplacian matrix by Lx = D
−1
2
x KxD
−1
2
x , where Dx
is a diagonal matrix of row sums of Kx. Similarly, Ly is computed for modality Y .
An important property of the Laplacian matrix is that its eigenvectors corresponding
to large eigenvalues reﬂect the underlying geometry of the data. The Laplacian
eigenvectors are used for many applications including data embeddings [28], clustering
[29], and feature selection [19]. For the latter, a popular metric for unsupervised
identiﬁcation of informative features is the Laplacian Score (LS) [19],
f TLxf =
n
X
i=1
λi(f Tui)2,
(1)
where Lx =
Pn
i=1 λiuiuT
i is the eigendecomposition of Lx and f is the normalized
feature vector.
Intuitively, when f varies slowly with respect to the underlying
structure of Lx, it will have a signiﬁcant component projected onto the subspace of
its top eigenvectors, and a higher score.
2.2
Diﬀerentiable Unsupervised Feature Selection
A key limitation of the Laplacian score stems from its underlying assumption that
the Laplacian matrix Lx accurately reﬂects the latent structure of the data. This
assumption, however, may not be valid in the presence of many noisy features. In such
cases, the top eigenvectors of Lx may be heavily inﬂuenced by noise and would not
capture the underlying structure accurately. A recent work [25] addresses this problem
by developing Diﬀerentiable Unsupervised Feature Selection (DUFS), a framework that
estimates the Laplacian matrix while simultaneously selecting informative features
using Laplacian scores. Speciﬁcally, DUFS computes a binary vector s ∈{0, 1}d that
indicates which features are kept (sj = 1) and which features are not (sj = 0). Let
∆(s) denote a diagonal matrix with s on the diagonal. At each iteration of DUFS,
the Laplacian is computed based on ˜
X = X∆(s), while simultaneously updaing s by
optimizing over the following loss function.
L = −1
nTr[ ˜
X
TL˜x ˜
X] + λ∥s∥0,
(2)
where Tr[] denotes the matrix trace. The ﬁrst term equals the sum of Laplacian Scores
across all features normalized by the total number of samples n in a training batch.
The second term is a ℓ0 regularizer that imposes sparsity to the number of selected
features, with λ being a tunable parameter that controls the sparsity level. The output
of DUFS is a list of a small number of selected features, and the Laplacian matrix L˜x
learned from them.
However, due to the discrete nature of the ℓ0 regularizer, the standard discrete
indicator vector s ∈{0, 1}D will make objective in Eq. (2) not diﬀerentiable and
5

## Page 6

ﬁnding the optimal solution intractable. Following, [23], one can relax the ℓ0 norm to a
probabilistic diﬀerentiable counterpart, by replacing the binary indicator vector s with
a relaxed Bernoulli vector z. Speciﬁcally, z is a continuous Gaussian reparametrization
of the discrete random variables, termed Stochastic Gates. It is deﬁned for each feature
i:
zi = max(0, min(1, 0.5 + µi + ϵi)),
ϵi ∼N(0, σ2)
(3)
where µi is a learnable parameter, and σ is ﬁxed throughout training. The loss function
in Eq. (2) can now be reformulated as follows, which is the ﬁnal objective of the
DUFS:
L = −1
nTr[ ˜
X
TL˜x ˜
X] + λ∥z∥0.
(4)
3
Method
We now derive our approach for unsupervised feature selection in multi-modal settings.
Our method is designed to capture two types of features: (i) Features associated with
latent structures that are shared between two modalities. (ii) Features associated with
diﬀerential latent structures, that appear in only one modality. In Sec. 3.1 and 3.2, we
derive two operators designed to capture shared and diﬀerential structures, respectively.
To motivate our approach and illustrate the diﬀerence between shared and diﬀerential
structures, we speciﬁcally address two examples: (i) shared and diﬀerential clusters
and (ii) product of manifolds. We use the proposed operators in Sec. 3.3 to derive
mmDUFS.
3.1
The shared structure operator
To motivate our approach, let us consider the artiﬁcial example illustrated in Fig.
2. The lower ﬁgure in the left panel shows the observations in modality Y , which
contains samples from a mixture of three distinct Gaussians. The upper ﬁgure shows
modality X, where one of the three clusters is partitioned again into three (less
distinct) clusters.
It is instructive to study the ideal setting where we make the following assumptions:
(i) The largest distance between two nodes within a cluster, denoted dwithin is much
smaller than the smallest distance between pairs of nodes of two clusters, denoted
dbetween. (ii) The bandwidth σx, σy is chosen such that dwithin ≪σx, σy ≪dbetween. In
this setting, the three Gaussians constitute three main clusters, with no connections
between pairs of nodes of diﬀerent clusters and similar weights between pairs of nodes
within clusters. Thus, the leading eigenvectors of Ly span the subspace of the three
indicator vectors. That is vectors that contain the square root of the degree of a node
in a cluster and a zero value outside the cluster. See [29] and illustration in Fig. 2.
6

## Page 7

Figure 2: Visualization of the eigenvectors and the aﬃnity matrix of the proposed
operators on an artiﬁcial cluster example. Left: Visualization of the clusters. Middle:
Leading eigenvectors of Lx and Ly. Right: Aﬃnity matrices of the proposed shared
graph operator (top) and the diﬀerential graph operator (bottom) with/without the
presence of noisy features.
The matrix Lx has two extra signiﬁcant eigenvectors that span the separation of the
third cluster, which appears only in X. We denote by V s a matrix that contains
the indicator vectors of the three partitions that appear in X and Y and by V x a
matrix that contains the partitions that appear only in X. In our ideal setting, the
two Laplacian matrices Lx, Ly are equal to
Lx ≈V sV T
s + V xV T
x ,
Ly ≈V sV T
s .
(5)
To capture shared latent structures we compute the following shared operator
P shared,
P shared = LxLy + LyLx.
(6)
For the cluster setting, the orthogonality between the matrices V s, V x implies
P shared ≈2V sV T
s .
The symmetric product of the two Laplacians captures clusters that appear in both
modalities while removing modality-speciﬁc clusters, see right panel of Fig. 2. We note
that a similar operator to Eq. (6) is proposed in [30] for computing low-dimensional
representations. Here, we combine our operator with DUFS to develop a multi-modal
feature selection pipeline. We illustrate the usefulness of the shared operator for the
product of manifold setting.
Product of manifolds.
Let Ma, Mb and Ms be three low-dimensional manifolds
embedded in Rn, which are smooth transformations of three sets of latent variables
θa, θb and θs. To further motivate our approach, consider the case where modalities
X and Y each contains observations from the products My, Mx given by,
My = Ms × Ma,
Mx = Ms × Mb.
7

## Page 8

Note that the dependence on θs is shared between Mx, My, while the dependence on
θa, θb is modality-speciﬁc.
In a product of manifolds Mx = Ms × Mb, every point x ∈Mx is associated
with two points xs ∈Ms and xb ∈Mb. Thus, we can deﬁne projection operators
πx
b (x), πx
s (x) that map a point x in Mx to points in Mb, Ms, respectively. In addition,
for every function f b : Mb →R we deﬁne its extension to the product manifold Mx
by
(f b ◦πx
b )(x) = f b(πx
b (x)).
An important property of a product Mx is that the eigenfunctions f x
l,m of the Laplace
Beltrami operator are equal to the pointwise product of the eigenfunctions of Mb, Ms,
extended to Mx.
f x
l,m = (f s
l ◦πx
s)(f b
m ◦πx
b ).
(7)
We refer to [31] for a detailed description of the properties of the product of manifolds.
A simple example of a product of manifolds is a 2D rectangle area (θs, θb) ∈[0, ls] ×
[0, lb]. the projection πx
s yields the ﬁrst coordinate, while πx
b yields the second. The
eigenfunctions of the product with Neumann boundary conditions are equal to,
fl,m = cos(πlθs/ls) cos(πmθb/lb).
(8)
Observations generated uniformly at random over the product of manifolds.
Here, we assume that the observations in the two modalities are generated by random
and independent uniformly distributed samples over Mx, My. Let φx
l,m(xi), φy
l,k(yi)
denote the eigenvectors of Lx, Ly evaluated at xi, yi respectively. In the asymp-
totic regime where the number of points n →∞, the eigenvectors converge to the
eigenfunctions as characterized in Eq. (7).
φx
l,m(xi) = φs
l (πx
s(xi))φb
m(πx
b (xi))
φy
l,k(yi) = φs
l (πy
s(yi))φa
k(πy
a(yi)).
(9)
Details about the deﬁnition and rate of convergence can be found, for example,
in [32, 33], and reference therein. It is instructive to consider the ideal case, where
due to their dependence on the independent projections πx
b and πx
a, the eigenvectors
φx
l,m, φy
l,k satisfy the following orthogonality property,
(φx
l,m)Tφy
l′,k =



1
l = l′, m = k = 0
0
o.w.
(10)
It follows that the operator P shared is equal to,
P shared = LxLy + LyLx =
X
l
(φs
l ⊗φa
0)(φs
l ⊗φb
0)T,
(11)
8

## Page 9

where ⊗denotes the Hadamard product. The vectors φa
0, φb
0 constitute the degree
of the diﬀerent observations and have little eﬀect on the outcome. Thus, the leading
eigenvectors of P shared are associated with the shared component and not the diﬀeren-
tial components in the product of manifolds. Below, we illustrate this phenomenon
with two examples.
Example 1: points in a 3D cube.
Consider points generated uniformly at random
over a 3D cube of dimensions [0, ls] × [0, la] × [0, lb]. Let Y ∈Rn×2 constitute the
ﬁrst two coordinates of n independent observations, and let X constitute the ﬁrst
and third coordinates. This is a simple case of a product of manifolds, where the
shared variable θs is the ﬁrst coordinate, while the modality-speciﬁc variables θa, θb
are the second and third coordinates. Following Eq. (8), the eigenvectors of the graph
Laplacian matrices Lx, Ly, evaluated at (θs, θb) and (θs, θa) converge to,
φx
lm(θs, θb) = cos(πlθs/ls) cos(πmθb/lb)
φy
lk(θs, θa) = cos(πlθs/ls) cos(πkθa/la).
(12)
The ﬁrst row of Fig. 1 (Appendix A) shows a scatter plot of the points in X (located
according to the ﬁrst two coordinates), colored by the values of the leading eigenvectors
of Lx. The second row shows the points in X, but colored by the eigenvectors of
P shared.
As expected, all the eigenvectors of P shared are functions of the shared
coordinate θs.
Example 2: videos taken from diﬀerent angles.
Our second example is based
on an experiment done in [34], where the two modalities constitute two videos of three
dolls rotating at diﬀerent angular speeds. The ﬁrst camera (modality X) captures the
middle and left doll, while the second camera (modality Y ) captures the middle and
right dolls (see Fig. 4a). Here, the shared variable θs is the angle of the middle doll
captured by both modalities. The modality-speciﬁc variables θa, θb are the angles of
the left and right dolls captured by each modality separately.
To illustrate Eq. (11) in this example, we ﬁrst compute an approximation of the
eigenvectors φs
l . To that end, we cropped each image in one of the videos such that only
the middle doll (which appears in both modalities) is shown. One may think of this
operation as a projection to the shared manifold. Next, we computed from the cropped
images the leading eigenvectors φs
l of the Laplacian matrix. Fig. 2 (Appendix A)
shows the leading three eigenvectors of P shared as a function of φs
1, φs
2, φs
3 as computed
by the cropped images. The ﬁgure shows a linear dependency between the vectors,
which implies that the shared operator retained only the shared component of the two
modalities.
9

## Page 10

3.2
The Diﬀerential Graph Operators
We design two operators Qx and Qy to infer latent structures that are modality speciﬁc
to X, Y respectively.
Qx = ˜L
−1
y Lx ˜L
−1
y ,
Qy = ˜L
−1
x Ly ˜L
−1
x ,
(13)
where ˜Lx = Lx +cI, ˜Ly = Ly +cI, and c is a regularization constant. We address the
cluster example used for the shared operator to motivate the use of these operators.
Diﬀerential clusters.
In the synthetic cluster example in Fig. 2, modality X has
three smaller clusters not observed in modality Y . We show that one can detect the
diﬀerential clusters of modality X via the leading eigenvectors of Qx. By Eq. (5), we
can approximate ˜Ly via,
˜Ly = (1 + c)V sV T
s + cV compV T
comp,
(14)
where V comp ∈Rn×(n−3) contains, as columns, vectors that span the complementary
subspace to V s. We write Qx as:
Qx = ˜L
−1
y Lx ˜L
−1
y
= c−2V xV T
x + (1 + c)−2V sV T
s .
(15)
The diﬀerential operator in Eq. (15) has two terms. The ﬁrst spans the subspace
corresponding to the diﬀerential structure V x, while the second spans the subspace of
the shared structure V s. Since c−2 > (1 + c)−2, it follows that the leading eigenvectors
of Qx span the subspace of V x.
In theory, we can directly apply these operators to learn the structures. However, in
many real-world applications, e.g., single-cell multi-omic technologies, both X and Y
can be very noisy. In particular, abundant noisy features (e.g., genes) might dominate
the data, and the top eigenvectors of Lx and Ly might not capture the underlying
structure, which would be detrimental to the learning of P shared, Qx, and Qy. As
shown in the aﬃnity matrices on the right of Fig. 2, the structures are less clear when
many noisy features are present. Therefore, it is necessary to have a feature selection
framework that can eﬀectively remove these noisy features in our multi-modal setting.
With the aforementioned DUFS feature selection framework as the foundation, we
will show in the next section how we can incorporate it into our proposed operators in
the multi-modal setting.
3.3
mmDUFS
In this section, we describe our framework, termed multi-modal Diﬀerential Unsuper-
vised Feature Selection (mmDUFS)1. We incorporates diﬀerentiable gates [25] with
1Codes are available at https://github.com/jcyang34/mmDUFS
10

## Page 11

loss functions based on the shared and diﬀerential operators, detailed in Sec. 3.1 and
3.2. Our goal is to compute an accurate shared graph operator (Pshared in Eq. (6)) and
diﬀerential graph operators (Qx and Qy in Eq. (13)) while simultaneously selecting
the informative features. Let f x, f y denote a feature vector in X, Y , respectively.
To quantify how noisy or informative the features are with respect to the shared
structure, we replace the Laplacian L in Eq. (1) with Pshared, which yields the shared
score f T
x Psharedf x and f T
y Psharedf y. Similarly, f T
x Qxf x and f T
y Qyf y quantify the
smoothness of these features with respect to the diﬀerential graph operators Qx and
Qy. The rationale behind these generalized Laplacian Scores is similar to the original
score. For instance, let Pshared = Pn
i=1 λiuiuT
i be the eigendecomposition of Pshared. If
f x varies slowly with respect to the underlying shared structure, it will have a larger
component projected onto the subspace of Pshared, thus leads to a higher score.
To learn features with high generalized Laplacian Scores and accurate graph
operators, mmDUFS learns two sets of Stochastic Gates zx and zy that ﬁlter irrelevant
features in each modality. Similar to DUFS [25], these stochastic gates multiply
the data matrices X and Y to remove nuisance features, i.e., ˜
X = X∆(zx) and
˜Y = Y ∆(zy). At each iteration, the updated graph operators ( ˜
Pshared, ˜
Qx, ˜
Qy) are
recomputed based on the gated inputs.
mmDUFS has two modes: (i) detecting shared structures using the shared graph
operator ˜
Pshared, and (ii) detecting modality-speciﬁc structures using the diﬀerential
graph operators ˜
Qx, and ˜
Qy. To learn the shared structure and the corresponding
features, we propose to optimize zx and zy by minimizing the following loss function:
Lshared = −1
nTr[ ˜
X
T ˜
Pshared ˜
X] −1
nTr[ ˜Y
T ˜
Pshared ˜Y ]
+ λx∥zx∥0 + λy∥zy∥0,
where the ﬁrst two terms are the Shared Laplacian Scores for each modality, and the
regularizers λx∥zx∥0 and λy∥zy∥0 control the number of selected features for each
modality, with tunable parameters λx, λy that control the level of sparsity. In Appendix
B.1, we suggest a procedure to tune these regularization parameters. Similarly, the
loss functions Lx, Ly are designed to detect features associated with structures that
appear only in modality X, Y , respectively.
Lx = −1
nTr[ ˜
X
TQ˜x ˜
X] + λx∥zx∥0,
Ly = −1
nTr[ ˜Y
TQ˜y ˜Y ] + λy∥zy∥0,
(16)
where the ﬁrst term in each loss is termed Diﬀerential Laplacian Scores. In the
following section we show the usefulness of these score functions for detecting relevant
features.
11

## Page 12

4
Results
We benchmark mmDUFS using synthetic and real multi-modal datasets. For discov-
ering the shared structures and associated features, we compare mmDUFS with the
shared operator to the following variants of kernel fusion-based methods previously
proposed for dimensionality reduction: (1) Matrix Concatenation (MC), where the
Laplacian is computed based on a concatenated matrix of the two modalities. (2)
Multi-modal Kernel Sum (mmKS) [35], where the Laplacian is equal to Lx + Ly. (3)
Multi-modal Kernel Product (mmKP) [36, 37]. where the Laplacian is equal to LxLy.
For each baseline, the k features with the highest Laplacian Scores are selected.
For the synthetic datasets, we set k to be the correct number of informative features.
We evaluate the performance of diﬀerent methods by the F1-score F1 = TP/(TP +
1
2(FP+FN)), where TP is the number of informative features selected by each method,
FP is the number of uninformative selected features, and FN is the number of
missed informative features. For the rescaled MNIST and rotating doll examples, the
informative features are set to the 25% pixels with the highest standard deviation.
4.1
Synthetic Examples
Rescaled MNIST.
We designed a rescaled MNIST example with shared and
modality-speciﬁc digits. We ﬁrst randomly sample one image (28 × 28 pixels) of digits
0, 3, 8. Then, we rescale each digit randomly and independently 500 times resulting
with 500 images of 0, 3, and 8. We concatenate pairs of 0 and 3 to create modality
X, and pairs of the same 3 and random 8 to create Y , see example in Fig. 3a. Thus,
this dataset consists of 500 samples and 28 × 56 pixels in each modality, with digit 3
shared between the modalities and digit 0 and 8 modality speciﬁc.
We apply mmDUFS with the shared operator to this example to select pixels
corresponding to 3. The left column of Fig. 3b shows the pixels gate values from
mmDUFS for modality X (top) and Y (bottom). We can see that selected pixels
outline the shape of the digit 3 well. Table 1 compares the F1-score achieved by
mmDUFS to three baselines. We can see that mmDUFS achieves a higher F1-score than
all the baselines on both modalities, demonstrating its ability to identify informative
features accurately.
Lastly, we apply mmDUFS with the diﬀerential operator to select modality-speciﬁc
pixels. The right column of Fig. 3b shows the pixel gate values for both modality
X (top) and Y (bottom). We can see that mmDUFS selects pixels that outline
digits 0, 8 for modalities X, Y , respectively. Additionally, mmDUFS achieves F1-score
0.8059 and 0.8832 for X and Y , showcasing its eﬀectiveness in identifying features
contributing to the diﬀerential structures.
12

## Page 13

(a)
(b)
(c)
(d)
(e)
Figure 3: Left (a-b): Evaluation of the proposed approach on the rescaled MNIST
dataset. (a): Random images from modality X (upper row) and modality Y (bottom
row) in gray-scale. (b): Selected pixels (dark blue) for the shared operator (left column)
and the diﬀerential operator (right column). Right (c-e): Synthetic developmental
tree example. (c): UMAP embeddings of the tree using data from modality X (top)
and modality Y (bottom). (d-e): Change of the Shared/Diﬀerential Laplacian Scores,
regularization loss, and the F1-score of the selected features concerning the number of
epochs (x-axis) for mmDUFS with the shared operator (panel (c)) and the diﬀerential
operator (panel (e)).
Dataset
Modality
MC
mmKS
mmKP
mmDUFS
Rescaled MNIST
X
0.3547
0.5291
0.5291
0.7093
Y
0.4826
0.6219
0.6219
0.8159
Synthetic Developmental Tree
X
0.6000
0.7800
0.8400
0.8800
Y
0.7800
0.8000
0.8200
0.9000
Original Gaussian
X
0.5000
0.7333
1
1
Y
0.5500
0.6500
0.9500
1
Gaussian + 10 Noisy Feats
X
0.5000
0.7333
1
1
Y
0.5000
0.6500
0.9000
1
Gaussian + 30 Noisy Feats
X
0.4667
0.7000
0.9667
1
Y
0.4500
0.5500
0.8500
1
Gaussian + 50 Noisy Feats
X
0.4000
0.6333
0.9333
0.9667
Y
0.4000
0.5500
0.8000
0.8500
Table 1: Comparison of F1-score between diﬀerent methods on the rescaled MNIST
example, the synthetic tree example, and the Gaussian mixture example with diﬀerent
numbers of additive noisy features.
Synthetic Developmental Tree.
Tree structures are ubiquitous throughout dif-
ferent biological processes and data modalities in single-cell biology [38, 39]. To
understand the interplay of diﬀerent mechanisms underlying the complex develop-
mental process, it is vital to discover the genetic features that contribute to the tree
structure shared across modalities and those that contribute to modality-speciﬁc
structures.
13

## Page 14

We evaluate mmDUFS using a simulated developmental tree example generated
via a tree simulator 2. The original data has 1000 samples and 100 features. We divide
the data into half, such that each modality has 50 informative features that contribute
to the shared tree structure, as shown in the UMAP embeddings in Fig. 3c, where
the samples in the tree are grouped into diﬀerent branch groups (labeled G1 to G6).
We then add 50 features drawn from negative binomial distributions to each modality
to create diﬀerential branches, that are only observed in one modality. Speciﬁcally,
branches G1 and G2 are bifurcated in modality X (top UMAP embeddings) but are
mixed in modality Y (bottom UMAP embeddings), and G3 and G4 are bifurcated in
modality Y but are mixed in modality X (see Supplementary section B.3 for further
details). After log transformation and z-scoring the data, we concatenate 200 features
drawn from N(0, 1) to each modality as noisy features.
We apply our model with the shared and diﬀerential operators to recover the
features that contribute to the overall tree structure and the set of features that
contribute to the split branches, respectively. Fig. 3d shows the change, during training
with the shared loss, in the Shared/Diﬀerential Laplacian Scores, the regularization
loss, and the F1-score. Fig. 3e shows the same properties for the diﬀerential loss.
Table 1 compares the F1-score of the selected features between diﬀerent methods.
Here as well, mmDUFS clearly outperforms the other methods.
(a)
(b)
(c)
(d)
(e)
Figure 4: Left (a-c): Rotating dolls example. (a): Random images of the dolls from
each video. (b-c): Selected pixels are marked in blue for mmDUFS with shared
operator (b) and the diﬀerential operator (c). Right (d-e): CITE-seq data example.
(d): UMAP embeddings using the RNA (top) and protein data (bottom), colored by
cell type labels. (e): Similar UMAP embeddings colored by the expression level of
several genes selected by mmDUFS with the diﬀerential operator.
2https://github.com/dynverse/dyntoy
14

## Page 15

Synthetic Gaussian Mixtures.
We generated a multi-modal Gaussian mixture
dataset, where X and Y each have 3 clusters. Two clusters are shared between
modalities, and cluster 3 and 4 are speciﬁc to X and Y , respectively. Each cluster
has a set of informative features drawn from a multivariate Gaussian, along with noisy
features (see Appendix B.2 for details).
We ﬁrst apply mmDUFS to uncover the informative features of the shared clusters
and the modality-speciﬁc clusters. In the ﬁgure of Supplementary section B.2, we
plot the change of the average shared/diﬀerential Laplacian Scores across features,
the regularization loss, and the F1-score of the selected features from mmDUFS with
respect to the number of epochs, where we can see that mmDUFS gradually selects
the correct features corresponding to high scores while sparsifying the number of
features. To evaluate mmDUFS’s feature selection capability in challenging regimes,
we further inject 10, 30, and 50 noisy features into each modality and compare the
F1-score of the selected features from diﬀerent methods in each regime. As shown in
Table 1, mmDUFS consistently outperforms the baseline methods while maintaining
accurate feature identiﬁcation capability, demonstrating its robustness against noise.
4.2
Real Data
Rotating Dolls.
We evaluate mmDUFS’s performance on the rotating doll video
dataset described in Sec. 3.1 in which 2 cameras capture 2 dolls from diﬀerent angles
(Fig. 4a). By treating each video frame as one sample (4050 in total) and the gray-
scaled pixels as features, we aim to uncover pixels that correspond to the shared doll
(the dog) and the modality-speciﬁc dolls (Yoda and rabbit).
For mmDUFS with the shared operator, Fig. 4b shows selected pixels in both
videos, as indicated by the blue dots. The shape of the dog is clearly delineated in both
modalities. We further compute the F1-score of the selected pixels with respect to the
underlying pixels that correspond to the dog. mmDUFS achieves F1-score of 0.7158
and 0.8033 for the two modalities, whereas MC achieves 0.2390 and 0.3822, and mmKS
and mmKP achieve 0.5452 and 0.6868. Fig. 4c shows the selected pixels of mmDUFS
with the diﬀerential operator in the two videos. In videos 1, mmDUFS select mostly
pixels corresponding to the Yoda (F1-score: 0.8861). For video 2, mmDUFS select
mostly pixels corresponding to the rabbit (F1-score: 0.7446).
CITE-seq Dataset.
In single-cell biology, cell states are characterized by diﬀerent
features at diﬀerent molecular levels. Identifying the contributing features is an open
question crucial to understanding the underlying cell systems. We apply mmDUFS to
a CITE-seq dataset from [3], in which cells are proﬁled at both transcriptomic and
proteomic levels measuring expressions of genes and protein markers, to identify the
genes and proteins that characterize the cell states in the multi-modal setting.
15

## Page 16

In this data, a group of murine cells is spiked-in as controls to human cord blood
mononuclear cells (CBMCs), and CITE-seq sequences the resulting cell system. Fig.
4d shows UMAP embeddings of the cells based on their RNA expression (top) and
protein expression (bottom). From the full dataset, we analyzed 3 cell populations:
murine cells (blue) and 2 CBMCs cell populations (Erythroids (orange) and CD34+
cells (green)). This dataset has 832 cells, with 500 top variable genes from modality
1 and 10 protein markers from modality 2. We can see that the murine cells are
separable from the Erythroids in the RNA space but not in the proteomic space. To
identify which gene markers contribute to the separation between cell groups, we apply
mmDUFS with the diﬀerential operator to this data. We found that all the selected
genes are murine genes that only express in the murine cells, as shown in Fig. 4e.
This example demonstrates that mmDUFS can identify genetic markers contributing
to the diﬀerential structures observed in single-cell multi-omic data.
5
Discussion
We present mmDUFS, a feature selection method that learns two novel graph operators
that capture the shared and the modality-speciﬁc structures in multi-modal data,
while simultaneously selecting the features that are informative for these structures.
MmDUFS can operate on small batches which makes it scalable to large datasets. On
the other hand, ﬁnding the optimal regularization parameters for mmDUFS on real
data may be challenging, for which we suggest an automatic procedure in Appendix
B.1. A second potential limitation is the O(n3) computational complexity required to
compute ˜L (Eq. (13)). A possible solution is to reduce the complexity by computing
a sparse Laplacian matrix.
Acknowledgements
The authors thank Amit Moscovich for the helpful discussions and feedback.
References
[1] Sai Ma, Bing Zhang, Lindsay M LaFave, Andrew S Earl, Zachary Chiang, Yan
Hu, Jiarui Ding, Alison Brack, Vinay K Kartha, Tristan Tay, et al. Chromatin
potential identiﬁed by shared single-cell proﬁling of rna and chromatin. Cell,
183(4):1103–1116, 2020.
16

## Page 17

[2] Yang Liu, Mingyu Yang, Yanxiang Deng, Graham Su, Archibald Enninful,
Cindy C Guo, Toma Tebaldi, Di Zhang, Dongjoo Kim, Zhiliang Bai, et al.
High-spatial-resolution multi-omics sequencing via deterministic barcoding in
tissue. Cell, 183(6):1665–1681, 2020.
[3] Marlon Stoeckius, Christoph Hafemeister, William Stephenson, Brian Houck-
Loomis, Pratip K Chattopadhyay, Harold Swerdlow, Rahul Satija, and Peter
Smibert. Simultaneous epitope and transcriptome measurement in single cells.
Nature methods, 14(9):865–868, 2017.
[4] Julia Joung, Sai Ma, Tristan Tay, Kathryn R Geiger-Schuller, Paul C Kirchgat-
terer, Vanessa K Verdine, Baolin Guo, Mario A Arias-Garcia, William E Allen,
Ankita Singh, et al. A transcription factor atlas of directed diﬀerentiation. Cell,
186(1):209–229, 2023.
[5] Yang Xiao, Graham Su, Yang Liu, Cheick A Sissoko, Yung-yu Huang, Adrienne N
Santiago, Andrew J Dwork, Gorazd B Rosoklija, Underwood D Mark, Victoria
Arango, et al. Spatially resolved transcriptomes in human hippocampus. Biological
Psychiatry, 91(9):S18, 2022.
[6] Noemie Leblay, Ranjan Maity, Elie Barakat, Sylvia McCulloch, Peter Duggan,
Victor Jimenez-Zepeda, Nizar J Bahlis, and Paola Neri. Cite-seq proﬁling of
t cells in multiple myeloma patients undergoing bcma targeting car-t or bites
immunotherapy. Blood, 136:11–12, 2020.
[7] Shiliang Sun. A survey of multi-view machine learning. Neural computing and
applications, 23:2031–2038, 2013.
[8] Xiaoqiang Yan, Shizhe Hu, Yiqiao Mao, Yangdong Ye, and Hui Yu.
Deep
multi-view learning methods: A review. Neurocomputing, 448:106–129, 2021.
[9] Tommi Raij, Kimmo Uutela, and Riitta Hari. Audiovisual integration of letters
in the human brain. Neuron, 28(2):617–625, 2000.
[10] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al. Learning transferable visual models from natural language supervision. In
International conference on machine learning, pages 8748–8763. PMLR, 2021.
[11] Andrey Guzhov, Federico Raue, Jörn Hees, and Andreas Dengel. Audioclip: Ex-
tending clip to image, text and audio. In ICASSP 2022-2022 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 976–980.
IEEE, 2022.
17

## Page 18

[12] Harold Hotelling. Relations between two sets of variates. Biometrika, 28(3/4):321–
377, 1936.
[13] Galen Andrew, Raman Arora, JeﬀBilmes, and Karen Livescu. Deep canonical
correlation analysis. In International Conference on Machine Learning, pages
1247–1255, 2013.
[14] Oﬁr Lindenbaum, Moshe Salhov, Amir Averbuch, and Yuval Kluger. L0-sparse
canonical correlation analysis. In International Conference on Learning Repre-
sentations, 2022.
[15] Harold Pimentel, Zhiyue Hu, and Haiyan Huang. Biclustering by sparse canonical
correlation analysis. Quantitative Biology, 6(1):56–67, 2018.
[16] Zhiwen Chen, Steven X Ding, Tao Peng, Chunhua Yang, and Weihua Gui.
Fault detection for non-gaussian processes using generalized canonical correlation
analysis and randomized algorithms. IEEE Transactions on Industrial Electronics,
65(2):1559–1567, 2017.
[17] Saúl Solorio-Fernández, J Ariel Carrasco-Ochoa, and José Fco Martínez-Trinidad.
A review of unsupervised feature selection methods. Artiﬁcial Intelligence Review,
53(2):907–948, 2020.
[18] Alexandra Degeest, Michel Verleysen, and Benoît Frénay. Smoothness bias in
relevance estimators for feature selection in regression. In Artiﬁcial Intelligence
Applications and Innovations: 14th IFIP WG 12.5 International Conference,
AIAI 2018, Rhodes, Greece, May 25–27, 2018, Proceedings 14, pages 285–294.
Springer, 2018.
[19] Xiaofei He, Deng Cai, and Partha Niyogi. Laplacian score for feature selection.
Advances in neural information processing systems, 18, 2005.
[20] Zheng Alan Zhao and Huan Liu. Spectral feature selection for data mining. Taylor
& Francis, 2012.
[21] Uri Shaham, Oﬁr Lindenbaum, Jonathan Svirsky, and Yuval Kluger.
Deep
unsupervised feature selection by discarding nuisance and correlated features.
Neural Networks, 152:34–43, 2022.
[22] Muhammed Fatih Balın, Abubakar Abid, and James Zou. Concrete autoencoders:
Diﬀerentiable feature selection and reconstruction. In International conference
on machine learning, pages 444–453. PMLR, 2019.
18

## Page 19

[23] Yutaro Yamada, Oﬁr Lindenbaum, Sahand Negahban, and Yuval Kluger. Feature
selection using stochastic gates. In International Conference on Machine Learning,
pages 10648–10659. PMLR, 2020.
[24] Junchen Yang, Oﬁr Lindenbaum, and Yuval Kluger.
Locally sparse neural
networks for tabular biomedical data. In International Conference on Machine
Learning, pages 25123–25153. PMLR, 2022.
[25] Oﬁr Lindenbaum, Uri Shaham, Erez Peterfreund, Jonathan Svirsky, Nicolas
Casey, and Yuval Kluger. Diﬀerentiable unsupervised feature selection based on
a gated laplacian. Advances in Neural Information Processing Systems, 34, 2021.
[26] George C Linderman, Manas Rachh, Jeremy G Hoskins, Stefan Steinerberger,
and Yuval Kluger. Fast interpolation-based t-sne for improved visualization of
single-cell rna-seq data. Nature methods, 16(3):243–245, 2019.
[27] Erez Peterfreund, Oﬁr Lindenbaum, Felix Dietrich, Tom Bertalan, Matan Gavish,
Ioannis G Kevrekidis, and Ronald R Coifman. Local conformal autoencoder for
standardized data coordinates. Proceedings of the National Academy of Sciences,
117(49):30918–30927, 2020.
[28] Mikhail Belkin and Partha Niyogi.
Laplacian eigenmaps for dimensionality
reduction and data representation. Neural computation, 15(6):1373–1396, 2003.
[29] Ulrike Von Luxburg. A tutorial on spectral clustering. Statistics and computing,
17(4):395–416, 2007.
[30] Tal Shnitzer, Mirela Ben-Chen, Leonidas Guibas, Ronen Talmon, and Hau-Tieng
Wu. Recovering hidden components in multimodal data with composite diﬀusion
operators. SIAM Journal on Mathematics of Data Science, 1(3):588–616, 2019.
[31] Sharon Zhang, Amit Moscovich, and Amit Singer. Product manifold learning. In
International Conference on Artiﬁcial Intelligence and Statistics, pages 3241–3249.
PMLR, 2021.
[32] Xiuyuan Cheng and Nan Wu. Eigen-convergence of gaussian kernelized graph
laplacian by manifold heat interpolation. Applied and Computational Harmonic
Analysis, 61:132–190, 2022.
[33] Nicolás García Trillos, Moritz Gerlach, Matthias Hein, and Dejan Slepčev. Error
estimates for spectral convergence of the graph laplacian on random geometric
graphs toward the laplace–beltrami operator. Foundations of Computational
Mathematics, 20(4):827–887, 2020.
19

## Page 20

[34] Roy R Lederman and Ronen Talmon.
Common manifold learning using
alternating-diﬀusion. submitted, Tech. Report YALEUIDCSITR1497, 2014.
[35] Dengyong Zhou and Christopher JC Burges. Spectral clustering and transductive
learning with multiple views. In Proceedings of the 24th international conference
on Machine learning, pages 1159–1166, 2007.
[36] Oﬁr Lindenbaum, Arie Yeredor, and Moshe Salhov. Learning coupled embedding
using multiview diﬀusion maps. In Latent Variable Analysis and Signal Separation:
12th International Conference, LVA/ICA 2015, Liberec, Czech Republic, August
25-28, 2015, Proceedings 12, pages 127–134. Springer, 2015.
[37] Oﬁr Lindenbaum, Arie Yeredor, Moshe Salhov, and Amir Averbuch. Multi-view
diﬀusion maps. Information Fusion, 55:127–149, 2020.
[38] Mireya Plass, Jordi Solana, F Alexander Wolf, Salah Ayoub, Aristotelis Misios,
Petar Glažar, Benedikt Obermayer, Fabian J Theis, Christine Kocks, and Nikolaus
Rajewsky. Cell type atlas and lineage tree of a whole complex animal by single-cell
transcriptomics. Science, 360(6391):eaaq1723, 2018.
[39] Kai Zhang, James D Hocker, Michael Miller, Xiaomeng Hou, Joshua Chiou,
Olivier B Poirion, Yunjiang Qiu, Yang E Li, Kyle J Gaulton, Allen Wang,
et al. A single-cell atlas of chromatin accessibility in the human genome. Cell,
184(24):5985–6001, 2021.
20

## Page 21

A
Additional Simulation Results
A.1
Points in a 3D cube.
The data consists of points in a 3D cube [0, ls] × [0, la] × [0, lb]. The modality X
includes the ﬁrst two coordinates, and modality Y includes the ﬁrst and third, as
explained in Sec. 3. The upper row in Figure A.1 shows the eigenvectors of Lx. The
eigenvectors change in both coordinates. The second row contains the eigenvectors
of P shared. the leading eigenvectors change only with the ﬁrst coordinate, as it is the
only shared variable.
Figure A.1: Data consists of points sampled uniformly at random in a 3D cube.
The upper row shows a scatter plot of the points, located according to the ﬁrst two
coordinates a, b and colored by the leading eigenvectors of Lx, the Laplacian matrix of
modality X. The bottom row shows the leading eigenvectors of P shared, the product
of Laplacians as deﬁned in Eq. 6.
A.2
Rotating Dolls.
The two modalities include video frames taken simultaneously from two cameras,
of three dolls rotating at diﬀerent angular speeds. The ﬁrst camera (modality X)
captures the left two dolls while the right camera (modality Y ) captures the right
two dolls. Thus, the angle of the middle doll constitutes a shared variable θs. The
angle of the left doll θx is modality X-speciﬁc latent variable, and the angle of the
right doll θy is modality Y -speciﬁc latent variable.
From the left video, we cut the frames such that it includes only the middle
doll (the shared component). From these images, we computed a graph Laplacian
21

## Page 22

matrix and its leading eigenvectors denoted φs
i. As explained in Sec. 3, we expect
the eigenvectors of the shared operator, denoted vs
i to be similar to φs
i, as both are
associated with the latent variable θs. Figure A.2 shows vs
i as a function of φs
i for
i = 1, 2, 3. The three vectors are clearly highly correlated.
Figure A.2: The ﬁgure shows a scatter plot of vs
i, the leading eigenvectors of Pshared
as a function of φs
i, the estimated leading vectors of the shared component in the
rotating doll dataset.
A.3
Synthetic Gaussian Mixtures.
Here we apply mmDUFS to uncover the informative features of the shared clusters
and the modality-speciﬁc clusters. Fig. A.3b and Fig. A.3c show the change of the
average Shared/Diﬀerential Laplacian Scores across features, the regularization loss,
and the F1-score of the selected features from mmDUFS with respect to the number
of epochs, where we can see that mmDUFS gradually selects the correct features
corresponding to high scores while sparsifying the number of features.
B
Experiment Details
In the following subsections, we provide additional experimental details required for
the reproduction of the experiments provided in the main text. The CPU model used
for the experiments is Intel(R) Xeon(R) Gold 6150 CPU @ 2.70GHz (72 cores total).
The GPU model is NVIDIA GeForce RTX 2080 Ti.
Below in Table B.1 and B.2, we list the parameters we used on each experiment
for mmDUFS with the shared operator and the diﬀerential operator. Parameter c is a
regularization constant for mmDUFS with the diﬀerential operator, as mentioned in
the main text. Parameter b is a scaling factor to the operators to balance between the
Shared/Diﬀerential Laplacian Scores with respect to the regularization term. We used
normalized Laplacian Matrix throughout the experiments except for the CITE-seq
22

## Page 23

(a)
(b)
(c)
Figure A.3: Synthetic Gaussian mixture cluster example. (a): Data matrix of modality
X (top) and Y (bottom). Rows are samples, and columns are features. Each modality
has 3 clusters (labeled in red). Clusters 1 and 2 are shared between modalities, and
cluster 3 and 4 are speciﬁc to each modality. (b): Change of the Shared Laplacian
Scores, regularization loss, and the F1-score of the selected features concerning the
number of epochs (x-axis) for mmDUFS with the shared operator. (c): Change of
the Diﬀerential Laplacian Scores, regularization loss, and the F1-score of the selected
features concerning the number of epochs (x-axis) for mmDUFS with the diﬀerential
operator.
example where we found the performance was satisfactory with the un-normalized
Laplacian Matrix.
Datasets
learning rate
epochs
λx
λy
b
Rescaled MNIST
2
10000
1e −1
1e −1
1e2
Synthetic Tree
2
25000
1e −1
1e −1
1e3
Gaussian Mixture
2
10000
1e −4
1e −4
1
Gaussian Mixture (10 Noisy Features)
2
20000
1e −8
1e −6
1
Gaussian Mixture (30 Noisy Features)
2
40000
1e −4
1e −4
1
Gaussian Mixture (50 Noisy Features)
2
10000
1e −2
1e −3
1e2
Rotating Dolls
2
10000
0.2
0.2
1e3
Table B.1: Parameters for mmDUFS with the shared operator across diﬀerent datasets.
For the baseline methods, k features with the highest Laplacian Scores are selected.
When evaluating the F1-score on the synthetic datasets, we set k to be the correct
23

## Page 24

Datasets
learning rate
epochs
λx
λy
c
b
Rescaled MNIST
1
10000
0.5
0.5
1e −3
1e −4
Synthetic Tree
2
10000
4
2
1e −3
1e −3
Gaussian Mixture
1
10000
0.4
0.4
1e −1
1e −1
Rotating Dolls
2
10000
2
2
3
1e3
CITE-seq
2
5000
3
2
1
Table B.2: Parameters for mmDUFS with the diﬀerential operator across diﬀerent
datasets.
number of informative features. To make a fair comparison, we also let mmDUFS
select k features by sorting the raw gates (µd for feature d). For other datasets, we
deﬁne selected features by mmDUFS as features whose gates converged to 1 (zd = 1
for feature d).
For the image datasets (rescaled MNIST, rotating dolls), we add small Gaussian
noise drawn from N(0, σ2) to the pixels to stabilize feature selection of mmDUFS. For
the rescaled MNIST dataset, σ = 0.1 and we add noise to the non-informative pixels
before standardizing the pixels via z-scoring. For the rotating dolls data, σ = 5e −3
and we add noise to all pixels before standardizing the pixels via z-scoring.
B.1
Tuning of the Regularization Parameter
mmDUFS has tunable regularization parameters λx and λy that control the sparsity of
the number of selected features. For synthetic datasets, one can tune these parameters
to select features such that the selected number is close to the prescribed number s.
However, it can still be time and resource-consuming to optimize these parameters.
Also, for real data, one might not know how many features to select and what λx and
λy to choose.
To alleviate this issue, we propose a "warm-up" procedure similar to [25] to
optimize λx and λy. Speciﬁcally, we evaluate the mean Shared Laplacian Scores
Sshared =
1
2n(Tr[ ˜
X
T ˜
Pshared ˜
X]/m+Tr[ ˜Y
T ˜
Pshared ˜Y ]/d) and the mean Diﬀerential Lapla-
cian Scores Sx = Tr[ ˜
X
TQ˜x ˜
X]/(d × n), Sy = Tr[ ˜Y
TQ˜y ˜Y ]/(m × n) over a grid of λx
and λy at the early stage of training (e.g., ﬁrst 1000 epochs), and pick the parameters
that maximize the Scores. Here n is the number of samples in the batch, and m and
d are the number of selected features on each modality for real data or the number of
pre-speciﬁed features for synthetic data.
To demonstrate this procedure, we use the synthetic Gaussian mixture dataset as
the example, and we evaluate λx and λy over {1e −6,1e −5,1e −4,1e −3,1e −2,1e −
24

## Page 25

Figure B.4: Evaluation of the mean Shared Laplacian Scores (left) and the correspond-
ing F1-scores (right) over a grid of λs on the synthetic Gaussian mixture dataset. the
y-axis shows the mean Shared Laplacian Scores (left) and F1-scores (right) whereas
the x-axis shows the values of λ.
1,1,1e1,1e2} using mmDUFS with the shared operator. For illustration purpose, we
set λx = λy Fig. B.4 shows the mean Shared Laplacian Scores over diﬀerent λ values.
We can see that {1e −6,1e −5,1e −4,1e −3} are the best candidates that give the
highest Shared Laplacian Scores that also correspond to the highest F1-score.
B.2
Synthetic Gaussian Mixtures
We simulate 2 modalities X and Y , where modality X has 260 samples with 130
features and modality Y has 260 samples with 90 features. Both modalities have 3
clusters in the data (X has cluster 1, 2, 3 and Y has cluster 1, 2, 4, all labeled in
red in Fig. A.3a), and each cluster has a set of informative features denoted as f x,i
and f x,i (i = 1, 2, 3, 4) with length mi (i = 1, 2, 3, 4). Each set of these informative
features is drawn from N(µi, I) independently for each sample, where µi is a vector
of length mi drawn from U(2, 4) and I is an mi × mi identity matrix.
By design, cluster 1 and 2 are shared between modalities with m1 = 20 and
m2 = 10 in modality X, and m1 = 10 and m2 = 10 in modality Y . On the other
hand, cluster 3 is speciﬁc to modality X with m3 = 40, and cluster 4 is speciﬁc to
modality Y with m4 = 40. The remaining features are considered noisy features and
are drawn from N(0, 1).
B.3
Synthetic Developmental Tree
We use generate_data() function from dyntoy 3,a tree simulator package, to gener-
ate a dataset X0 with 1000 samples and 100 features. Speciﬁcally, the parameter
3https://github.com/dynverse/dyntoy
25

## Page 26

num_branchpoints is set to 1, num_cells is set to 1000, num_features is set to
100, sample_mean_count is set to 10, sample_dispersion_count is set to 50, dif-
ferentailly_expressed_rate is set to 4, and dropout_probability_factor is set to 0.
This step yields an initial data matrix X0 ∈R1000×100, and these 1000 samples are
initially partitioned into 4 groups: G1 and G2, G3 and G4, G5, G6 shown in Fig. 3c.
For X0, we further divide it into two halves, resulting in 2 data matrices X ∈R1000×50
and Y ∈R1000×50. We regard X and Y as 2 data modalities and these features as
informative features contributing to the shared tree structure.
We further add 50 features to each modality that are drawn from negative binomial
distributions to construct the diﬀerential structures between modalities. Speciﬁcally,
for modality X, the 50 features of G1 are drawn from NB(µ = 4, α = 0.1) where µ and
α are the mean and dispersion parameter of the negative binomial distribution, whereas
the 50 features of the other groups of samples are drawn from NB(µ = 20, α = 0.1).
Similarly, for modality Y , the 50 features of G3 are drawn from NB(µ = 4, α = 0.1)
while the 50 features of the other groups of samples are drawn from NB(µ = 20, α =
0.1). Therefore, G1 is bifurcated from G2 and this structure is only observed in X,
and G3 is bifurcated from G4 and this structure is only observed in Y .
Next, we row normalize each data matrix with a scaling factor 1e4, and log1p
transform the data. Then we standardize the features by z-scoring. At the end, we
add 200 features drawn from N(0, 1) to each modality as the noisy features.
B.4
CITE-seq
The human cord blood mononuclear cells (CBMCs) CITE-seq data was generated by
[3], where the expression levels of both RNA and protein are measured for the same
cells. We analyze 3 cell types: Erythroid cells, CD 34+ cells, and Murine cells. We
row normalize each data matrix for both modalities. For the gene expression matrix
(RNA), we ﬁlter the genes by standard deviation and keep the top 500 variable genes.
Then for both matrices, we standardize the features by z-scoring.
26
