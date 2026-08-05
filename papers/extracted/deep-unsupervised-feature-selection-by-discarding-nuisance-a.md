---
source_pdf: papers/Deep Unsupervised Feature Selection by Discarding Nuisance and Correlated Features.pdf
slug: deep-unsupervised-feature-selection-by-discarding-nuisance-a
pages: 16
extracted_on: 2026-08-04
---

# Deep Unsupervised Feature Selection by Discarding Nuisance and Correlated Features

## Page 1

Deep Unsupervised Feature Selection by Discarding Nuisance
and Correlated Features
Uri Shaham∗, Oﬁr Lindenbaum∗, Jonathan Svirsky, Yuval Kluger
October 2021
Abstract
Modern datasets often contain large subsets of correlated features and nuisance features,
which are not or loosely related to the main underlying structures of the data. Nuisance features
can be identiﬁed using the Laplacian score criterion, which evaluates the importance of a given
feature via its consistency with the Graph Laplacians’ leading eigenvectors. We demonstrate
that in the presence of large numbers of nuisance features, the Laplacian must be computed on
the subset of selected features rather than on the complete feature set. To do this, we propose
a fully diﬀerentiable approach for unsupervised feature selection, utilizing the Laplacian score
criterion to avoid the selection of nuisance features. We employ an autoencoder architecture
to cope with correlated features, trained to reconstruct the data from the subset of selected
features. Building on the recently proposed concrete layer that allows controlling for the number
of selected features via architectural design, simplifying the optimization process. Experimenting
on several real-world datasets, we demonstrate that our proposed approach outperforms similar
approaches designed to avoid only correlated or nuisance features, but not both. Several state-
of-the-art clustering results are reported. Our code is publicly available at https://github.
com/jsvir/lscae.
1
Introduction
Feature Selection is an important area of machine learning. Reducing data dimensionality may be
appealing for numerous reasons, for example, reduction of the computational burden, overﬁtting
prevention, and simpliﬁcation of downstream tasks and analyses, to name a few. In the supervised
learning regime, it is straightforward to evaluate the quality of diﬀerent subsets of input features
simply by measuring the generalization performance of a model given the selected features as inputs.
Evaluation of feature selection in the unsupervised regime, on the other hand, is a more involved
task, as there is no natural (single) criterion for evaluation of the quality of diﬀerent subsets of
selected features.
We identify two dominant feature evaluation criteria in the unsupervised feature selection lit-
erature. The ﬁrst criterion is the consistency of a feature with the main underlying structures of
the data. It is common to associate these main underlying structures with the leading eigenvectors
of the graph Laplacian matrix of the data.
It is well known that when the data is clusterable,
the cluster structure can be recognized in the subspace of Laplacian’s leading eigenvectors [15]. In
addition, when the data has a low-dimensional structure (e.g., lies on a manifold), the diﬀusion
∗equal contribution
1
arXiv:2110.05306v1  [stat.ML]  11 Oct 2021

## Page 2

distance [3], which describes the similarity between data points, is governed by the large eigenvalues
of the Laplacian.
Thus, a common means to measure the consistency of a feature with the main underlying data
structures is by the inner product of the given feature with the leading eigenvectors of the graph
Laplacian of the data [6, 2, 20, 17], which is known as the “Laplacian score” criterion. Thus, the
Laplacian score criterion favors features that have signiﬁcant components in the subspace of the
leading Laplacian eigenvectors. Selection of features that respect the multi-cluster structure was
also approached via alternative measures; see, for example [18].
The second feature evaluation criterion is the amount of information a feature contains on other
features in the dataset. The logic behind this criterion is that a subset of representative features is
helpful for selection if it carries a suﬃcient amount of information to represent the complete feature
set, despite being sparse. Thus, a common means to achieve this goal is to search for a subset of
features from which the full feature set can be approximately reconstructed. Such an approach can
be found in [21, 1, 16, 5]. It is easy to see how this criterion can yield a highly sparse set of selected
pixels on the MNIST handwritten dataset, for example, while capturing suﬃcient information to
reconstruct all pixels, as neighboring pixels in this dataset tend to be highly correlated.
Each of the two above criteria implicitly deﬁnes an inductive bias, which speciﬁes the character-
istics of features that should be selected. The ﬁrst criterion favors features which correspond to the
main structures (e.g., cluster structures or “slow-varying” ones) in the data, hence tends to discard
features which do not manifest this structure; such features are often called “nuisance features”, as
opposed to “informative features”, which do carry information on the data underlying structures.
The second criterion favors features that are correlated with a large number of other features and
can thus be used to reconstruct the complete feature set approximately.
Realistically, however, modern datasets contain both types of features (i.e., nuisance ones and
correlated ones), making any feature selection method that is designed to handle only one type
of features sub-optimal 1. Therefore combining the two evaluation criteria can arguably be more
suited for general unsupervised feature selection purposes. Despite that, to the best of the authors’
knowledge, most of the existing unsupervised feature selection methods are not designed to handle
both correlated and nuisance features.
Moreover, while the Laplacian score is a widely used tool for evaluating feature importance, it
is commonly computed based on a Laplacian that relies on the complete feature set. However, in
the presence of a large number of nuisance features, the Laplacian gets corrupted in the sense that
its leading eigenvectors no longer correspond to the manifold or cluster structure of the data. We
demonstrate this problematic aspect of the Laplacian score. We argue that the Laplacian should
be computed on the subset of selected features to avoid it rather than on the complete feature
set. While this makes the selection process more involved, it can be stated as a diﬀerentiable cost
function; hence it can be solved using tools commonly used in deep learning.
Being a convenient framework for optimizing diﬀerentiable objective functions, deep learning
algorithms are widely used for many traditional machine learning tasks. This is the case also in
unsupervised feature selection, where several recently proposed methods are implemented as deep
learning algorithms. For example ,[1, 16, 5] are autoencoder-based methods, which aim to reconstruct
the data from a small subset of selected features, hence correspond to the second criterion. [12] is
based on stochastic gates and aims to ﬁnd a small subset of selected features that maximize the
Laplacian score, hence corresponds to the ﬁrst criterion.
In this work, we follow this direction and propose a deep learning method for unsupervised
1In this sense, a comparison between methods corresponding to diﬀerent criteria, often found in the unsupervised
feature selection literature, may be comparing apples and oranges, as the diﬀerences in performance can depend more
on the types of features which exist in the dataset on which the comparison is made, and less on algorithmic matters.
2

## Page 3

feature selection, which aims to correspond to the two above criteria by discarding both nuisance and
correlated features in a diﬀerential fashion. Our method builds on the recently proposed Concrete
Autoencoder (CAE, [1]), augmenting its objective function using a diﬀerential Laplacian score term.
CAE is equipped with a Concrete layer, which controls the number of selected features using an
elegant architectural design; we utilize this mechanism to tackle the problematic aspects of computing
the Laplacian score for the complete feature set.
To highlight the utility of the proposed approach, we ﬁrst simulate a scenario in which it stands
out compared to related methods that handle either nuisance or correlated features, but not both,
in the sense that it allows better recognition of the data true cluster structure. We then report
experimental results on ten real-world datasets, demonstrating that the proposed approach can lead
to state-of-the-art downstream clustering tasks.
Our contributions are four-fold: (i) We demonstrate that a large number of nuisance features
corrupts the Laplacian, making the Laplacian score a sub-optimal measure for feature quality; we also
provide analytical arguments supporting this claim. (ii) We propose an autoencoder-based approach
for unsupervised feature selection, designed to handle both correlated and nuisance features. (iii)
We experimentally demonstrate the advantage of the proposed approach over methods that consider
either correlated features or nuisance ones, but not both, on several real-world unsupervised feature
selection benchmarks, reporting state-of-the-art performance. (iv) We provide a user-friendly Python
implementation of the proposed approach for general use.
This work builds on an earlier work of ours Lindenbaum et al. [12], which contains only the
Laplacian score objective and utilizes stochastic gates for the selection mechanism. In particular
Section 3 appears in similar form therein. Yet, the proposed methodology in the current manuscript
diﬀers signiﬁcantly from the one in [12].
The remainder of this manuscript is organized as follows. In Section 2 we review preliminary
materials. Our proposed approach is motivated in Section 3 and described in Section 4. Experimental
results are provided in Section 5. Section 6 brieﬂy concludes the manuscript.
2
Preliminaries
Consider a data matrix X ∈Rn×d with d-dimensional observations x1, . . . , xn. We refer to the
columns of X as features f1, . . . , fd, and we assume that the features are centered and normalized,
i.e., for each i, 1T fi = 0 and ∥fi∥2
2 = 1.
2.1
Laplacian score
Given n data points, a kernel matrix is a n × n matrix K, whose (i, j) entry quantiﬁes the similarity
between xi and xj. For example, in many applications, such matrix is constructed using a Gaussian
kernel
Ki,j = exp

−∥xi −xj∥2
2σ2

,
where σ is a user-deﬁned parameter that determines the sensitivity of the kernel2.
Given a kernel matrix K, the unnormalized graph Laplacian Lun is deﬁned via Lun = D −K,
where D is a diagonal matrix of row sums of K. It is common to interpret the Laplacian eigenvalues
as frequencies so that eigenvectors corresponding to larger eigenvalues oscillate faster [3]. Assuming
that important underlying patterns of the data (e.g., cluster structure) are slowly varying, the
2A common practice to choose σ is to set it to the maximal Euclidean distance from any point to its nearest
neighbor, and many other practices exist as well.
3

## Page 4

eigenvectors corresponding to the smallest eigenvalues of Lun express the main structures of the data.
This fact is used as a basis for various manifold learning and dimensionality reduction techniques.
For example, when the Laplacian represents clusterable data with m distinct components, the
leading m eigenvectors provide a complete speciﬁcation of the cluster allocation of any point; this
insight led to the celebrated spectral clustering method[15]. Since the leading eigenvectors of the
Laplacian describe the important structures of the data, it makes sense to evaluate a feature by how
much it respects this structure. This idea lies at the core of the Laplacian score method, proposed
by [6], where each feature f is assigned the score
score(f) = f T Lunf =
n
X
i=1
λi⟨ui, f⟩2,
where Lun = Pn
i=1 λiuiuT
i is the eigendecomposition of Lun. Therefore, the smaller the score a
feature f is assigned, the more signiﬁcant is the component of f in the subspace of the leading
eigenvectors of Lun, implying that f is more consistent with the main structures of the data, making
it an essential feature for selection.
Similar behavior of the eigenvectors of the unnormalized Laplacian exists for the diﬀusion Lapla-
cian Ldiﬀ= D−1K which expresses the transition probabilities between any pair of points, except
that for Ldiﬀthe eigenvectors corresponding to largest eigenvalues are the ones that express the main
structures in the data. To use the same terminology for both Laplacians, we use the term “leading”
to refer to the eigenvectors corresponding to the smallest eigenvalues of unnormalized Laplacian and
the eigenvectors corresponding to the largest eigenvectors in case of the diﬀusion Laplacian.
2.2
Concrete Layer
The Concrete distribution [13] is a continuous relaxation of discrete random variables, which allows
diﬀerentiation through a sampling procedure, in a similar fashion to the reparametrization trick
for continuous random variables [9]. This opens a wide range of deep learning applications which
incorporate discrete random variables into the training procedure.
More speciﬁcally, a sample z of a categorical random variable with probabilities (π1, . . . , πd) can
be obtained via the Gumble-max trick [4]
z = arg max
i (gi + log πi),
(1)
where g1, . . . , gd are iid samples from a Gumble(0, 1) distribution. Since the softmax function is a
continuous approximation of the arg max() function, equation (1) can be relaxed into a continuous
approximation [8] via
zi =
exp((gi + log πi)/τ)
Pd
j=1 exp((gj + log πj)/τ)
,
(2)
where τ is a temperature parameter, governing the extent to which the softmax vector is peaked.
In a concrete layer, each unit approximates the sampling of a single entry from its input vector.
The approximation is performed by a dot product of the Gumble-Softmax vector (2) with the input.
The temperature τ is typically annealed throughout training, starting from a high value (for which
the resulting softmax vector is ﬂat) and ending in a value near zero, for which the softmax vector is
close to being one-hot. In the latter case, the dot product approximates sampling from a categorical
distribution. The probabilities πi, i = 1, . . . , d of each unit are learnable diﬀerentiable parameters,
which are trained via backpropagation. A concrete layer of size k is therefore parametrized by a
k ×d parameter Π, where Πi,1, . . . Πi,d are the categorical probabilities of the ith concrete unit. This
results in an k × d feature selection weight matrix Z, obtained via equation (2).
4

## Page 5

2.3
Concrete Autoencoder
Concrete Autoencoder (CAE, [1]) is a state-of-the-art deep unsupervised feature selection method,
based on a standard autoencoder, having a concrete layer as its ﬁrst layer. The size of the concrete
layer determines the desired number of selected features. The autoencoder is trained by minimizing
reconstruction loss using gradient based optimization. As explained in the previous section, after
the temperature annealing process, each unit in the concrete layer approximates sampling of a single
input feature from a categorical distribution, which amounts to selecting a single feature.
The reconstruction error loss makes CAE select features from which the full feature set can be
reconstructed. Hence CAE is designed for scenarios where there are subsets of correlated features,
where each such subset can be reconstructed from a single or a few representative features.
The concrete layer of CAE provides an elegant, architectural-based control for the number of
selected features, which simpliﬁes the training process, as the loss function can contain only the
reconstruction error term. This alleviates the need for a regularization term to encourage sparsity
of the selected subset, as in the core of several other deep learning approaches for feature selection,
e.g., [12].
3
Motivation
This section motivates the need to compute the graph Laplacian on the subset of selected features
rather than on the complete feature set when the data contains many nuisance features. To do so,
we ﬁrst present a diﬀusion perspective and demonstrate the change in the Laplacian eigenvalues and
eigenvectors empirically as the number of nuisance features grows. We then consider a two-cluster
case study and show analytically how the number of nuisance dimensions aﬀects the ability to recover
the main underlying structure of the data.
3.1
A Diﬀusion Perspective
Consider the simple 2-dimensional dataset, known as “Two moons”, shown in the top left panel of
Figure 1, which contains two (nonconvex) separate clusters. We extend this dataset by adding k
nuisance dimensions, each of which is a sample of iid unif(0, 1) entries. As the number k of nuisance
dimensions grows, the clear cluster structure is obscured, as the amount of noise dominates the
signal. Consequently, recognizing the actual underlying two-cluster structure becomes challenging
and is likely to fail.
From a diﬀusion perspective, data are considered clusterable if a random walk that starts inside
a cluster takes a long time to exit the cluster. The cluster exit times are manifested by the leading
(i.e., largest) eigenvalues of the diﬀusion Laplacian Ldiﬀ= D−1W (for example, when the data
contains m completely separate clusters, and a random walk can never leave the cluster it begins
at, the top m eigenvalues of Ldiﬀare all equal to 1). Each added nuisance dimension increases
the variability inside any cluster and increases the distances between any point and its true nearest
neighbors (that is, ones which belong to the same “moon” in the two-moons example). At the same
time, the added noise is likely to create spurious similarities between points, regardless of the actual
cluster they belong to. Altogether, this shortens the cluster exit times, which is manifested by the
decrease of the second largest eigenvalue of Ldiﬀ, as is shown in the top right panel of Figure 1. A
similar behavior occurs by looking at the second smallest eigenvalue of the unnormalized Laplacian
Lrw = D −W, known as the “algebraic connectivity” or Fiedler number, which increases with the
number of nuisance dimensions, implying that these dimensions make the graph more connected, as
is shown in Figure 1 as well (middle left panel). The fact that the graph becomes more connected
5

## Page 6

is also manifested by the second leading eigenvector of the Laplacian, which becomes less indicative
of the correct cluster assignment as the number of nuisance dimensions grows (middle right panel).
As a result of the graph becoming more connected, attempts to recover the actual cluster structure
are more likely to fail as the number k of nuisance dimensions grows. One may argue that this
can be avoided by using a dimensionality reduction technique, like PCA, to capture the signal (i.e.,
the correct cluster structure) and remove much noise. However, as can be seen in the bottom left
panel of Figure 1, using PCA in this example, unfortunately, does not capture the actual underlying
structure, as the directions of maximal variance correspond to noise and not to the correct cluster
structure. Applying our proposed approach to the above dataset, the two informative dimensions
are selected, and the nuisance dimensions are discarded. As a result, downstream algorithms like
SpectralNet [14] can correctly identify the cluster structure (bottom right panel).
To complement this empirical analysis, in the next section, we consider a simple two-cluster case.
We analytically derive the connection between the number of nuisance dimensions and one’s ability
to recover the cluster structure of the data.
3.2
Case Study Analysis
To observe the eﬀect of nuisance dimensions, in this section, we consider a simple example where all
of the noise in the data arises from such dimensions. Speciﬁcally, consider a dataset that includes 2n
datapoints in R, where n of which are at 0 ∈R and the remaining ones are at r > 0, i.e., each cluster
is concentrated at a speciﬁc point. Next, we add d nuisance dimensions to the data so that samples
lie in Rd+1. The value for each data point in each nuisance dimension is sampled independently
from N(0, 0.52).
Suppose we construct the graph Laplacian by connecting each point to its nearest neighbors.
We would now investigate the conditions under which the neighbors of each point belong to the
correct cluster. Consider points x, y belonging to the same cluster. Then (x −y) = (0, u1, . . . , ud)
where ui
iid
∼N(0, 1), and therefore ∥x −y∥2 ∼χ2
d. Similarly, if x, y belong to diﬀerent clusters, then
∥x −y∥2 ∼r2 + χ2
d. Now, to ﬁnd conditions for n and d under which with high probability the
neighbors of each point belong to the same cluster, we can utilize Chi-square measure-concentration
bounds [10].
Lemma 3.1 ([10] P.1325). Let X ∼χ2
d. Then
1. P(X −d ≥2√dγ + 2γ) ≤exp(−γ).
2. P(d −X ≥2√dγ) ≤exp(−γ).
Given suﬃciently small γ > 0 we can divide the segment [d, d + r2] to two disjoint segments of
lengths 2√dγ + 2γ and 2√dγ (and solve for d in order to have the total length r2). This yields
√
d = r2 −2γ
4√γ
.
(3)
The nearest neighbors of each point will be from the same cluster as long as all distances between
points from the same cluster will be at most d + 2√dγ + 2γ and all distances between points from
diﬀerent clusters will be at least d + r2 −2√dγ. According to lemma 3.1, this will happen with
probability at least (1 −exp(−γ))2n2−n. Denoting this probability as 1 −ϵ and solving for γ, we
obtain
γ ≤−log(1 −
(2n2−n)√
1 −ϵ).
(4)
6

## Page 7

Figure 1: Top left: the Two moons dataset. Top right: the second largest eigenvalue of the diﬀusion
Laplacian decreases as the number of nuisance dimensions grows. Middle left: similarly, the algebraic
connectivity of the graph increases with more nuisance dimensions. Middle right: with more nuisance
dimensions, the second leading eigenvector of the graph Laplacian is no longer a clear indicator of
the cluster assignments. In each subplot, the vertical position corresponds to the entry in the second
leading Laplacian eigenvector; the horizontal position corresponds to the cluster assignment. Bottom
left: Projection of the data on the ﬁrst two principal directions. PCA cannot recover the underlying
cluster structure, as the directions of maximal variance correspond to noise. Bottom right: our
proposed approach identiﬁes the informative dimensions, which enables downstream analysis, e.g.,
using SpectralNet [14].
Plugging (4) into (3) we obtain
d = O

r4
−log(1 −
(2n2−n)√1 −ϵ)

.
(5)
In particular, for ﬁxed n and ϵ, equation (5) implies that the number of nuisance dimensions
must be at most on the order of r4 for the clusters to not mix with high probability. In addition,
increasing the number of data points for a ﬁxed r and ϵ brings the argument inside the log term
arbitrarily close to zero, which implies that the Laplacian for large data is sensitive to the number
7

## Page 8

of nuisance dimensions. We support these ﬁndings via experiments, as shown in Figure 2.
Figure 2: Synthetic two cluster datasets. We evaluate the inﬂuence of Gaussian nuisance variables on
the Laplacian. We generate two clusters using 50 samples each with distance r apart in 1-D. We use
d Gaussian nuisance variables and evaluate the leading nontrivial eigenvector ψ2 of the Laplacian.
Left: correlation between ψ2 and the true cluster assignments y for diﬀerent values of r. As the
number of nuisance variables grows, the eigenvector becomes meaningless. As the distance between
clusters decreases, fewer nuisance variables are needed to “break” the cluster structure captured by
ψ2. Right: by computing the intersection between the damped correlation curves and 0.7 (shown in
the left plot) for diﬀerent values of r, we evaluate the relation between r and the number of nuisance
variables d required for breaking the cluster structure. This empirical result supports the analysis
presented in 3.1 in which we show that d = O

r4
−log(1−(2n2−1)√1−ϵ)

. For convenience, we added a
polynomial ﬁt up to degree 4 presented as the black line.
4
The Proposed Approach
In this section, we present our proposed approach and discuss several of its characteristics.
4.1
Rational
In section 3 we demonstrated the problems in computing the Laplacian score using the entire feature
set. We concluded that the Laplacian score should ideally be calculated using a Laplacian that is
not aﬀected by many nuisance features. Here we show that this can be tackled by computing the
Laplacian score at the CAE concrete layer to achieve a feature selection mechanism that discards
both nuisance and correlated ones.
As explained in Section 2.3, CAE is an autoencoder model, equipped with a concrete layer
in which, at the end of the training, each concrete unit simulates sampling from a categorical
distribution with learnable class probabilities.
At the beginning of training, the softmax vector
of each concrete unit tends to be ﬂat due to high temperature, as can be seen in equation (2).
Propagating the data through the concrete layer and using this data representation to compute the
Laplacian score creates a corrupted Laplacian, in which the entire feature set, including nuisance
features, is taken into account. While this is undesirable, as explained above, the contribution of
informative features to the Laplacian score often tends to be slightly higher than the contribution
of nuisance features, as is noticeable in the left panel of ﬁgure 3. We utilize this fact to create a
8

## Page 9

learning dynamic that promotes the selection of informative (i.e., not nuisance) features by adding
the Laplacian score to the CAE objective function. Doing so encourages the sampling probabilities
of informative features to grow, and this dynamic strengthens during training, as can be seen in
the middle and left panels of ﬁgure 3.
As is the case for CAE, at the end of the training, the
temperature is low, which results in the concrete softmax vectors being approximately one-hot,
which eﬀectively simulates a feature selection mechanism. Hence, by computing the Laplacian score
at the CAE concrete layer and adding it to the CAE objective function, one obtains a feature selection
mechanism biased towards selecting (i) informative features, which (ii) suﬃce for the approximate
reconstruction of the complete feature set.
Figure 3: Two moons dataset: contribution f T Lf of each feature f to the Laplacian score. The two
leftmost coordinates represent informative (structured) features in all subplots, and the remaining
are nuisance features. Left: when the Laplacian is computed using all feature sets, the informative
coordinates contribute more to the Laplacian score, which initiates the learning dynamics. Middle:
during training, the sampling in the concrete layer gives greater probabilities to sample from the
informative coordinates. As a result, the Laplacian score grows. Right: when only informative,
structured coordinates are sampled, the Laplacian score is maximized.
4.2
LS-CAE
In this manuscript, the approach we propose, termed LS-CAE for Laplacian Score-regularized CAE,
is an extension of CAE, essentially by adding a Laplacian score term to its objective function during
training, where the Laplacian is computed at the concrete layer.
Speciﬁcally, the proposed approach inherits from CAE the autoencoder framework and the con-
crete layer and the reconstruction loss as a feature selection mechanism that promotes selection of
a sparse subset of representative features which capture much of the information of the complete
feature set. As this alone does not encourage the discarding of nuisance features, we augment the
CAE objective function with a Laplacian score term. However, armed with the insight that the
Laplacian should be computed on the selected features rather than on the complete feature set, we
calculate the Laplacian at the concrete layer.
Experimentally, we have noticed that the two objective losses might be of very diﬀerent magni-
tudes at diﬀerent times during training, resulting in a single term dominating the training dynamics.
To avoid this, we utilize a balancing mechanism that ensures that the two objective components are
of similar magnitude and can both aﬀect the training dynamic and the selected features.
More formally, let X be a m × d minibatch of training data. Denote by ˆX the autoencoder
output, and denote the output of the concrete layer by C = C(X). Our proposed objective function
is therefore
L(X) =
∥X −ˆX∥2
2
SG

∥X −ˆX∥2
2
 −
Trace[CT Ldiﬀ(C)C]
SG (Trace[CT Ldiﬀ(C)C]),
(6)
9

## Page 10

where Ldiﬀ(C) is the diﬀusion Laplacian D−1W, computed on the concrete layer representation of
the data, and SG is the Stop Gradient operator, which acts as an identity at forward and has zero
partial derivatives.
The balancing mechanism, whose magnitude inversely weights each loss component, removes the
need to use a tunable hyperparameter to balance between them and ensures both terms are taken
into account in selecting features. In addition, we have empirically observed that this results in a
more stable training dynamic, comparing to [12], where the Laplacian score term alone encourages
the selection of all features. In contrast, the regularization term encourages the opposite goal, as
the opposition forces may result in instability of the training process and increased sensitivity to
hyper-parameter tuning.
4.3
Penalizing Redundant Selection of Same Feature
The concrete layer mechanism allows scenarios where two or more concrete units select the same
input feature. While this is wasteful from a reconstruction perspective, the Laplacian score term
might beneﬁt from it. To avoid this, we add a regularization term, penalizing the selection of a
feature more than once. This regularization term is computed as follows
reg = M max{0, m −1},
(7)
where M is a large constant, m is the maximal sum of weights of any features by concrete units, i.e.,
m := max
j=1,...d
k
X
i=1
Zij,
(8)
and Z is the k × d-sized matrix of concrete layer probabilities.
4.4
Temperature Annealing
As the diﬀerence in the contributions to the Laplacian score between nuisance and informative
features might be small at the beginning of training when the Laplacian is corrupted (for example,
in the left panel of Figure 3), it is beneﬁcial to let this term undergo a warm-up period at the
beginning of training before the diﬀerence between a nuisance and important features starts to
become apparent through the concrete probabilities.
To allow for this warm-up period, we use a linearly decaying temperature annealing schedule,
rather than the exponential schedule initially used in the oﬃcial code of [1]3. Eﬀectively, the slower
temperature annealing schedule enables various subsets of selected features to be evaluated during
training before the probabilities settle on sampling from the desired features. Experimentally we
noticed that without changing the annealing schedule, in the absence of the “warm-up period”, the
Laplacian score term was sometimes unable to avoid the selection of nuisance features.
5
Experimental Results
This section provides experimental results on simulated and real-world datasets, demonstrating the
proposed approach’s eﬃcacy compared to other unsupervised feature selection baselines. We begin
with Ablative experiments, justifying the proposed design, and then to real-world data experiments.
3https://github.com/mfbalin/Concrete-Autoencoders
10

## Page 11

5.1
Ablation Study
Our proposed objective function (6) contain a reconstruction term and a Laplacian score term. In
this section, we design two simulated experiments. We show that the proposed objective yields a
better selection of features compared to methods containing one of the objective terms, but not
both. Speciﬁcally, we compare our approach (LS-CAE), concrete autoencoder (CAE), which utilizes
only the reconstruction term, and a model using only the Laplacian score objective (LS). All models
shared an identical architecture and training hyperparameter conﬁguration.
5.1.1
Simulated Data
In this experiment, we construct the dataset as follows: The dataset contains n = 1200 and 2d + 4
input features, where 2 of the features are the original two moons features, as in the top left panel of
Figure 1. We then add another noisy copy of the two original features and two copies of d nuisance
features, obtained by sampling a multivariate d-dimensional multivariate Gaussian, with zero mean
and covariance C such that Cij = (−0.25)|i−j|. All models were trained with two concrete units.
We measured the proportion of times where the two selected input features were the two original
two moon coordinates (from either of the two copies).
The dataset was constructed this way to demonstrate that in the presence of correlated features:
(i) the reconstruction term alone might favor features which allow for low reconstruction error,
despite being high-frequency (i.e., irrelevant to the cluster structure) and (ii) the Laplacian score
term might favor two copies of the same original input feature and ignore other low-frequency
features. The dataset was z-transformed before training the model so that all features had zero
mean and unit standard deviation.
We trained the model for d = 3, 6, 12, 15, with ten repetitions per setting (where repetitions diﬀer
in the sampling of the dataset and the initialization of the model). Figure 4 shows the results of this
experiment. As can be seen, having both objective terms consistently (overall values of d) yields the
selection of better features than just one of the objectives but not the other objective in this case.
Figure 4: Ablation study: augmented two moons data.
11

## Page 12

5.1.2
MNIST
In this experiment we create a noisy version of the MNIST handwritten digit dataset, via replacing
xi with min{0, max{255, xi + np ⊙mi}}, where np is a 28 × 28 noise pattern sampled a i.i.d uniform
distribution over {0, 1, . . . , 255}, mi is a 28 × 28 i.i.d Bernoulli(0.2) mask and ⊙denotes element-
wise product. As the noise pattern, np is common to all images; the data contains correlated high-
frequency features, which might lead to the sub-optimal selection of features using the reconstruction
term alone. In addition, since adjacent pixels are typically highly correlated in the MNIST datasets,
the Laplacian score term might favor selecting such pixels while ignoring other other features that
are important to identify the image type. Examples of noisy images are shown in Figure 5
Figure 5: Examples from the noisy mnist dataset.
We trained the models to select 5, 10, 15, 20 and 25 features. Once the features were selected, we
trained a k-means with k = 10 on the training dataset (60,000 examples), using only the selected
features, and measured the clustering accuracy on the test dataset (10,000 examples). For each
number of features, we repeated the above procedure three times.
The results are presented in
Figure 6, which shows the average clustering accuracy of each of the methods, and also the clustering
accuracy obtained when the features are selected randomly.
Figure 6: Results of the noisy MNIST experiment.
12

## Page 13

As evident from this plot, the proposed approach consistently leads to higher clustering accuracies
(compared to the other baselines). This suggests that it can identify a better subset of features that
carry information of the main underlying structure in the data. Figure 7 shows examples of the
features selected by each of the methods. As can be seen, the added noise makes CAE select pixels
Figure 7: Noisy MNIST experiment: example of 15 features selected by each of the methods.
scattered over a large portion of the image, many of which are not indicative of the digit type. About
half of the features selected by the Laplacian score term are near the boundary of the image, as they
are low-frequency but irrelevant for identifying the digit type. On the other hand, LS-CAE seems
to select pixels concentrated in areas of the images relevant for determining the cluster component
the image belongs to.
5.2
Real world datasets
To demonstrate the advantage of using LS-CAE on real-world data, we now turn our attention to
nine publicly available feature selection benchmark datasets 4. Table 1 summarizes the properties
of the datasets used for these experiments. The datasets vary in sample size, from as few as 56
examples to 21,332. In addition, they contain a large number of features, often higher than the
sample size. On such datasets, the quality of the set of selected features can dramatically improve
the performance of downstream tasks, such as clustering, as will be demonstrated next. The k-means
clustering accuracy on each dataset using all features (i.e., without any feature selection) is indicated
in Table 1 as well.
Datasets
Dim
Samples
Classes
Accuracy using All features
RCV1
24408
21332
2
50.0
GISETTE
4955
6000
2
74.4
PIX10
10000
100
10
74.3
COIL20
1024
1444
20
53.6
Yale
1024
165
15
38.3
TOX-171
5748
171
4
41.5
ALLAML
7192
72
2
67.3
PROSTATE
5966
102
2
58.1
FAN
25683
56
8
37.5
POLLEN
21810
301
4
54.9
Table 1: Properties of each of the real-world benchmark datasets used in the experiments.
We compare to several strong baselines, such as Laplacian Score [6] (LS), Multi-Cluster Feature
4https://jundongl.github.io/scikit-feature/datasets.html
13

## Page 14

Selection [2] (MCFS), Local Learning based Clustering (LLCFS) [19], Nonnegative Discriminative
Feature Selection (NDFS) [11], Multi-Subspace Randomization and Collaboration (SRCFS) [7], and
Concrete Auto-encoders (CAE) [1]. Each model is tuned to select the best 50, 100, 150, 200, 250, or
300 features. Then, we apply k-means 20 times on the selected features and compute the average
clustering accuracy. For each method, we report the highest (average) clustering accuracy along
with the number of selected features.
Datasets
LS
MCFS
NDFS
LLCFS
SRCFS
CAE
LS-CAE
RCV1
54.9 (300)
50.1 (150)
55.1 (150)
55.0 (300)
53.7 (300)
54.9 (300)
83.7 (300)
GISETTE
75.8 (50)
56.5 (50)
69.3 (250)
72.5 (50)
68.5 (50)
77.3 (250)
80.7 (50)
PIX10
76.6 (150)
75.9 (200)
76.7 (200)
69.1 (300)
75.9 (100)
94.1 (250)
94.5 (250)
COIL20
60.0 (300)
59.7 (250)
60.1 (300)
48.1 (300)
59.9 (300)
65.6 (200)
61.8 (300)
Yale
42.7 (300)
41.7 (300)
42.5 (300)
42.6 (300)
46.3 (250)
45.4 (250)
48.0 (200)
TOX-171
47.5 (200)
42.5 (100)
46.1 (100)
46.7 (250)
45.8 (150)
47.7 (100)
48.3 (100)
ALLAML
73.2 (150)
72.9 (250)
72.2 (100)
77.8 (50)
67.7 (250)
73.5 (250)
76.5 (150)
PROSTATE
58.6 (300)
57.3 (300)
58.3 (100)
57.8 (50)
60.6 (50)
56.9 (250)
71.4 (50)
FAN
42.9 (150)
45.5 (150)
48.8 (100)
29.0 (50)
29.0 (100)
35.2 (300)
51.7 (100)
POLLEN
46.9 (150)
66.5 (300)
48.9 (50)
35.0 (100)
34.9 (300)
58.0 (250)
65.8 (100)
Mean rank
4.0
6.0
5.0
5.0
6.0
2.0
1.0
Median rank
3.67
5.89
4.33
4.67
5.22
2.89
1.33
Table 2: Average clustering accuracy on several benchmark datasets. Clustering is performed by
applying k-means to the features selected by the diﬀerent methods. The number of selected features
is shown in parenthesis.
As can be seen, on seven of the ten benchmark datasets, LS-CAE achieves the best perfor-
mance, and the second-best on the remaining three, with an average rank of 1. Noticeably, on the
RC1 benchmark, our proposed approach outperforms the following best method by 50%. On the
Prostate and FAN dataset, our proposed approach also signiﬁcantly improves more than 10% over
the following best methods.
5.3
Technical details
All the experiments reported in this manuscript were performed using the same decoder architecture,
containing two hidden layers, each of 128 LeakyReLU units. We trained the models for 300 epochs,
using a learning rate of 1. for the concrete layer and 0.01 for the decoder.
6
Conclusion
In this manuscript, we propose an unsupervised approach for feature selection, utilizing concrete layer
mechanism and Laplacian score and reconstruction objectives. We demonstrated both analytically
and numerically that for the Laplacian score to be a helpful criterion in the presence of many
nuisance features, it is crucial to be computed on the subset of selected features rather than on the
complete feature set. We showed that the proposed objective function’s two components are needed
to avoid both high frequency and correlated features and reported state-of-the-art results on several
real-world datasets of various sizes.
References
[1] Balın, M. F., Abid, A., and Zou, J. (2019).
Concrete autoencoders: Diﬀerentiable feature
selection and reconstruction. In International Conference on Machine Learning, pages 444–453.
14

## Page 15

[2] Cai, D., Zhang, C., and He, X. (2010). Unsupervised feature selection for multi-cluster data.
In Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and
data mining, pages 333–342.
[3] Coifman, R. R. and Lafon, S. (2006). Diﬀusion maps. Applied and computational harmonic
analysis, 21(1):5–30.
[4] Gumbel, E. J. (1954). Statistical theory of extreme values and some practical applications: a
series of lectures, volume 33. US Government Printing Oﬃce.
[5] Han, K., Wang, Y., Zhang, C., Li, C., and Xu, C. (2018). Autoencoder inspired unsupervised fea-
ture selection. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP), pages 2941–2945. IEEE.
[6] He, X., Cai, D., and Niyogi, P. (2006). Laplacian score for feature selection. In Advances in
neural information processing systems, pages 507–514.
[7] Huang, D., Cai, X., and Wang, C.-D. (2019). Unsupervised feature selection with multi-subspace
randomization and collaboration. Knowledge-Based Systems, 182:104856.
[8] Jang, E., Gu, S., and Poole, B. (2017). Categorical reparametrization with gumble-softmax. In
International Conference on Learning Representations (ICLR 2017). OpenReview. net.
[9] Kingma, D. P. and Welling, M. (2013).
Auto-encoding variational bayes.
arXiv preprint
arXiv:1312.6114.
[10] Laurent, B. and Massart, P. (2000). Adaptive estimation of a quadratic functional by model
selection. Annals of Statistics, pages 1302–1338.
[11] Li, Z., Yang, Y., Liu, J., Zhou, X., and Lu, H. (2012). Unsupervised feature selection using
nonnegative spectral analysis. In Twenty-Sixth AAAI Conference on Artiﬁcial Intelligence.
[12] Lindenbaum, O., Shaham, U., Svirsky, J., Peterfreund, E., and Kluger, Y. (2020).
Let
the data choose its features:
Diﬀerentiable unsupervised feature selection.
arXiv preprint
arXiv:2007.04728.
[13] Maddison, C. J., Mnih, A., and Teh, Y. W. (2016). The concrete distribution: A continuous
relaxation of discrete random variables. arXiv preprint arXiv:1611.00712.
[14] Shaham, U., Stanton, K., Li, H., Nadler, B., Basri, R., and Kluger, Y. (2018). Spectralnet:
Spectral clustering using deep neural networks. arXiv preprint arXiv:1801.01587.
[15] Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and computing, 17(4):395–
416.
[16] Wang, S., Ding, Z., and Fu, Y. (2017). Feature selection guided auto-encoder. In Thirty-First
AAAI Conference on Artiﬁcial Intelligence.
[17] Wang, S., Tang, J., and Liu, H. (2015). Embedded unsupervised feature selection. In Proceedings
of the AAAI Conference on Artiﬁcial Intelligence, volume 29.
[18] Yang, Y., Shen, H. T., Ma, Z., Huang, Z., and Zhou, X. (2011). ell 2, 1-norm regularized
discriminative feature selection for unsupervised learning. In IJCAI international joint conference
on artiﬁcial intelligence.
15

## Page 16

[19] Zeng, H. and Cheung, Y.-m. (2010). Feature selection and kernel learning for local learning-
based clustering. IEEE transactions on pattern analysis and machine intelligence, 33(8):1532–
1547.
[20] Zhao, Z. and Liu, H. (2007). Spectral feature selection for supervised and unsupervised learning.
In Proceedings of the 24th international conference on Machine learning, pages 1151–1157.
[21] Zhu, P., Zuo, W., Zhang, L., Hu, Q., and Shiu, S. C. (2015). Unsupervised feature selection by
regularized self-representation. Pattern Recognition, 48(2):438–446.
16
