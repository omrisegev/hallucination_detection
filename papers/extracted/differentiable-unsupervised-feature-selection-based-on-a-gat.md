---
source_pdf: C:/Users/omris/TAU/hallucination_detection/papers/Differentiable Unsupervised Feature Selection based on a Gated Laplacian.pdf
slug: differentiable-unsupervised-feature-selection-based-on-a-gat
pages: 21
extracted_on: 2026-09-07
---

# Differentiable Unsupervised Feature Selection based on a Gated Laplacian

## Page 1

Diﬀerentiable Unsupervised Feature Selection based on a
Gated Laplacian
Oﬁr Lindenbaum 1∗
Uri Shaham 1∗
Jonathan Svirsky 2
Erez Peterfreund3
Yuval Kluger1†
1Yale University, USA; 2 Technion, Israel; 3 Hebrew University, Israel
†Corresponding author. E-mail: yuval.kluger@yale.edu
Address: 333 Cedar St, New Haven, CT 06510, USA
∗These authors contributed equally.
Abstract
Scientiﬁc observations may consist of a large number of variables (features). Identifying
a subset of meaningful features is often ignored in unsupervised learning, despite its
potential for unraveling clear patterns hidden in the ambient space. In this paper, we
present a method for unsupervised feature selection, and we demonstrate its use for the
task of clustering. We propose a diﬀerentiable loss function which combines the Laplacian
score, that favors low frequency features, with a gating mechanism for feature selection.
We improve the Laplacian score, by replacing it with a gated variant computed on a
subset of features. This subset is obtained using a continuous approximation of Bernoulli
variables whose parameters are trained to gate the full feature space. We mathematically
motivate the proposed approach, and demonstrate that in the high noise regime, it is
crucial to compute the Laplacian on the gated inputs, rather than on the full feature set.
Experimental demonstration of the eﬃcacy of the proposed approach and its advantage
over current baselines is provided using several real-world examples.
1
Introduction
Recently there has been a growing interest in the machine learning community towards un-
supervised and self-supervised learning. This was motivated by impressive empirical results
demonstrating the beneﬁts of analyzing large amounts of unlabeled data (for example in natural
language processing). In many scientiﬁc domains, such as biology and physics, the growth of
computational and storage resources, as well as technological advances for measuring numerous
features simultaneously, makes the analysis of large, high dimensional datasets an important
research need. In such datasets, discarding irrelevant (i.e. noisy and information-poor) features
may reveal clear underlying natural structures that are otherwise hidden in the high dimensional
space. We refer to these uninformative features as “nuisance features”. While nuisance features
are mildly harmful in the supervised regime, in the unsupervised regime discarding such features
is crucial and may determine the success of downstream analysis tasks (e.g., clustering or
manifold learning). Some of the pitfalls caused by nuisance features could be mitigated using
an appropriate unsupervised feature selection method.
1
arXiv:2007.04728v3  [cs.LG]  9 Nov 2020

## Page 2

The problem of feature selection has been studied extensively in machine learning and
statistics. Most of the research is focused on supervised feature selection, where identifying
a subset of informative features has beneﬁts such as reduction in memory and computations,
improved generalization and interpretability. Filter methods, such as [1, 2, 3, 4, 5, 6] attempt
to remove irrelevant features prior to learning a model. Wrapper methods [7, 8, 9, 10, 11] use
the outcome of a model to determine the relevance of each feature. Embedded methods, such as
[12, 13, 14, 15, 16] aim to learn the model while simultaneously select the subset of relevant
features.
Unsupervised feature selection methods mostly focus on two main tasks: clustering and
dimensionality reduction or manifold learning. Among studies which tackle the former task, [17,
18, 19] use autoencoders to identify features that are suﬃcient for reconstructing the data. Other
clustering-dedicated unsupervised feature selection methods asses the relevance of each feature
based on diﬀerent statistical or geometric measures. Entropy, divergence and mutual information
are used in [20, 21, 22, 23, 24] to identify features which are informative for clustering the data.
A popular tool for evaluating features is the graph Laplacian [25, 26]. The Laplacian Score (LS)
[27], evaluates the importance of each feature by its ability to preserve local structure. The
features which most preserve the manifold structure (captured by the Laplacian) are retained.
Several studies, such as [28, 29, 30], extend the LS based on diﬀerent spectral properties of the
Laplacian.
While these methods are widely used in the feature selection community, they rely on the
success of the Laplacian in capturing the “true” structure of the data. We argue that when
the Laplacian is computed based on all features, it often fails to identify the informative ones.
This may happen in the presence of a large number of nuisance features, when the variability of
the nuisance features masks the variability associated with the informative features. Scenarios
like this are prevalent in areas such as bioinformatics, where a large number of biomarkers
are measured to characterize developmental and chronological biology processes such as cell
diﬀerentiation or cell cycle. These processes may depend merely on a few biomarkers. In these
situations, it is desirable to have an unsupervised method that can ﬁlter nuisance features prior
to the computation of the Laplacian.
In this study, we propose a diﬀerentiable objective for unsupervised feature selection. Our
proposed method utilizes trainable stochastic input gates, trained to select features with high
correlation with the leading eigenvectors of a graph Laplacian that is computed based on
these features. This gating mechanism allows us to re-evaluate the Laplacian for diﬀerent
subsets of features and thus unmask informative structures buried by the nuisance features. We
demonstrate, that the proposed approach outperforms several unsupervised feature selection
baselines.
2
Preliminaries
Consider a data matrix X ∈Rn×d, with d dimensional observations x1, ..., xn. We refer to
the columns of X as features f 1, ..., f d, where f i ∈Rn and assume that features are centered
and normalized such that 1Tf i = 0 and ∥f i∥2
2 = 1. We assume that the data has an inherent
structure, determined by a subset of the features S∗and that other features are nuisance
variables. Our goal is to identify the subset of relevant features S∗and discard the remaining
ones.
2

## Page 3

2.1
Graph Laplacians
Given n data points, a kernel matrix is a n × n matrix K so that Ki,j represents the similarity
between xi and xj. A popular choice for such matrix based on the Gaussian kernel
Ki,j = exp
 
−∥xi −xj∥2
2σ2
b
!
,
where σb is a user-deﬁned bandwidth (chosen, for example, based on the minimum values of
the 1-nearest neighbors of all points). The unnormalized graph Laplacian matrix is deﬁned as
Lun = D −K, where D is a diagonal matrix D, whose elements Di,i = Pn
j=1 Ki,j correspond
to the degrees of the points i = 1, ..., n. The random walk graph Laplacian is deﬁned as
Lrw = D−1K, and expresses the transition probabilities of a random walk to move between data
points. Graph Laplacian matrices are extremely useful in many unsupervised machine learning
tasks. In particular, it is known that the eigenvectors corresponding to the small eigenvalues of
the unnormalized Laplacian (or the large eigenvalues of the random walk Laplacian) are useful
for embedding the data in low dimension (see, for example, [31]).
2.2
Laplacian Score
Following the success of the graph Laplacian [26] and [25], the authors in [27] have presented an
unsupervised measure for feature selection, termed Laplacian Score (LS). The LS evaluates each
feature based on its correlation with the leading eigenvectors of the graph Laplacian.
At the core of the LS method, the score of feature f is determined by the quadratic form
f TLf, where L = Lun is the unnormalized graph Laplacian. Since
f TLf =
n
X
i=1
λi⟨ui, f⟩2,
where L = Pn
i=1 λiuiuT
i is the eigen-decomposition of L, the score is smaller when f has a
larger component in the subspace of the smallest eigenvectors of L. Such features can be
thought of as “informative”, as they respect the graph structure. Eigenvalues of the Laplacian
can be interpreted as frequencies, and eigenvectors corresponding to larger eigenvalues of Lun
(or smaller eigenvalues of Lrw) oscillate faster. Based on the assumption that the interesting
underlying structure of the data (e.g. clusters) depends on the slowly varying features in the
data, [27] proposed to select the features with the smallest scores.
2.3
Stochastic Gates
Due to the enormous success of gradient decent-based methods, most notably in deep learning,
it is appealing to try to incorporate discrete random variables into a diﬀerentiable loss functions
designed to retain the slow varying features in the data. However, the gradient estimates of
discrete random variables tend to suﬀer from high variance [34]. Therefore, several authors have
proposed continuous approximations of discrete random variables [35, 36]. Such relaxations have
been used for many applications, such as model compression [37], discrete softmax activations
[38] and feature selection [13]. Here, we use a Gaussian-based relaxation of Bernoulli variables,
termed Stochastic Gates (STG) [13] which is diﬀerentiated based on the repamaterization trick
[32, 33].
We denote the STG random vector by Z ∈[0, 1]d, parametrized by µ ∈Rd, where each entry
is deﬁned as
Zi = max(0, min(1, µi + ϵi)),
(1)
3

## Page 4

Figure 1: Left: The stochastic gate Z is deﬁned via the repamaterization trick [32, 33]. Standard
Gaussian noise is injected and shifted by a trainable parameter µi, the result is thresholded to
[0, 1] based on (1). Two examples of the density of the stochastic gate Zi. Middle: at initialization
µi = 0.5 and the gate approximates a ’fair’ Bernoulli variable. Right: the distribution at µi = −1
approximates a ’closed’ gate.
where µi is a learnable parameter, ϵi is drawn from N(0, σ2) and σ is ﬁxed throughout training.
This approximation can be viewed as a clipped, mean-shifted, Gaussian random variable. In
Fig. 1 we illustrate the gating mechanism, and show examples of the densities of Zi for diﬀerent
values of µi.
Multiplication of each feature by its corresponding gate enables us to derive a fully dif-
ferentiable feature selection method. At initialization µi = 0.5, i = 1, ..., d, so that all gates
approximate a ”fair” Bernoulli variable. The parameters µi can be learned via gradient decent
by incorporating the gates in a diﬀrentiable loss term. To encourage feature selection in the
supervised setting, [13] proposed the following diﬀerentiable regularization term
r(Z) =
d
X
i=1
P(Zi ≥0) =
d
X
i=1
 1
2 −1
2 erf
 
−µi
√
2σ
!!
,
(2)
where erf() is the Gauss error function. The term (2) penalizes open gates, so that gates
corresponding to features that are not useful for prediction are encouraged to transition into a
closed state (which is the case for small µi).
3
Demonstration of the Importance of Unsupervised Fea-
ture Selection in High Dimensional Data with Nui-
sance Features
We ﬁrst demonstrate the importance of feature selection in unsupervised learning when the data
contains nuisance variables, by taking a diﬀusion perspective. Then, we use a simple two cluster
model to analyze how Gaussian nuisance dimensions aﬀect clustering capabilities.
3.1
A Diﬀusion Perspective
Consider the following 2-dimensional dataset, known as “two-moons”, shown in the top-left panel
of Fig. 2. We augment the data with k “nuisance” dimensions, where each such dimension is
composed of i.i.d unif(0, 1) entries. As one may expect, when the number of nuisance dimensions
is large, the amount of noise (manifested by the nuisance dimensions) dominates the amount of
signal (manifested by the two “true” dimensions). Consequently, attempts to recover the true
structure of the data (say, using manifold learning or clustering) are likely to fail.
4

## Page 5

Figure 2: Two-moons experiment. Top Left: the original 2-dimensional dataset. Top right:
the second largest eigenvalue of the random walk matrix Lrw decreases as the number k of
nuisance dimensions grows. This implies that the graph becomes more connected, and hence
less clusterable, as the number of nuisance dimensions grows. Bottom left: the second largest
eigenvector ψ2 (y-axis) of Lrw becomes less representative of the true cluster structure as k
grows. Bottom right: projecting the data onto the leading two principal directions (x and y
axes) cannot recover the true cluster structure when k > 0.
From a diﬀusion perspective, data is considered to be clusterable when the time it takes a
random walk starting in one cluster to transition to a point outside the cluster is long. These exit
times from diﬀerent clusters are manifested by the leading eigenvalues of the Laplacian matrix
Lrw = D−1K, for which the large eigenvalues (and their corresponding eigenvectors) are the
ones that capture diﬀerent aspects of the data’s structure (see, for example, [39]). Each added
nuisance dimension increases the distance between a point and its “true” nearest neighbors
along one of the “moons”. In addition, the noise creates spurious similarities between points,
regardless of the cluster they belong to. Overall, this shortens the cluster exit times. This
phenomenon can be captured by the second largest eigenvalue λ2 of Lrw (the largest eigenvalue
λ1 = 1 carries no information as it corresponds to the constant eigenvector ψ1), which decreases
as the number of nuisance dimensions grows, as shown in the top right panel of Fig. 2. The fact
that λ2 decreases implies that the diﬀusion distances [39] in the graph decrease as well, which in
turn means that the graph becomes more connected, and hence less clusterable. A similar view
may be obtained by observing that the second smallest eigenvalue of the un-normalized graph
Laplacian Lun = D −W , also known as Fiedler number or algebraic connectivity, grows with
the number of nuisance variables. The fact that the graph becomes less clusterable as more
nuisance dimensions are added is also manifested by the eigenvector ψ2 corresponding to the
second largest eigenvalue of Lrw (or the second smallest eigenvalue of Lun), which becomes less
representative of the cluster structure (bottom left panel of Fig. 2).
Altogether, this means that in order for the data to be clusterable, the noisy features ought
to be removed. One may argue that principal component analysis can be used to retain the
5

## Page 6

Figure 3: Demonstrating the information captured by the Laplacian Score on the noisy-two
moons dataset (see Fig. 2). The ﬁrst two features are informative, while the rest of the variables
are nuisance. Our goal is to train the diﬀerentiable stochastic gates for identifying the two
informative features. Left: Laplacian score f TLf at initialization, based on all 10 dimensions in
total. The score for the informative features is slightly higher. Middle: Laplacian score, based
on the gated Laplacian f TLXS∗f at convergence of the gates. The informative features attain a
substantial higher score based on the gated Laplacian. Right: the parameter-free loss (black
line) and the average number of active gates (red line) as a function of the number of epochs.
signal features while removing the noise. Unfortunately, as shown in the bottom right panel
of Fig. 2, projecting the data onto the ﬁrst two principal directions does not yield the desired
result, as the variance along the noise directions is larger than along the signal ones. In the
next sections we will describe our diﬀerentiable unsupervised feature selection approach, and
demonstrate that it does succeed to recognize the important patterns of the data in this case.
3.2
Analysis of Clustering with Nuisance Dimensions
In order to observe the eﬀect of nuisance dimensions, in this section we consider a simple
example where all of the noise in the data arises from such dimensions. Speciﬁcally, consider
a dataset that includes 2n datapoints in R, where n of which are at 0 ∈R and the remaining
ones are at r > 0, i.e., each cluster is concentrated at a speciﬁc point. Next, we add d nuisance
dimensions to the data, so that samples lie in Rd+1. The value for each datapoint in each
nuisance dimension is sampled independently from N(0, 0.52).
Suppose we construct the graph Laplacian by connecting each point to its nearest neighbors.
We would now investigate the conditions under which the neighbors of each point belong to the
correct cluster. Consider points x, y belonging to the same cluster. Then (x−y) = (0, u1, . . . , ud)
where ui
iid∼N(0, 1), and therefore ∥x −y∥2 ∼χ2
d. Similarly, if x, y belong to diﬀerent clusters,
then ∥x −y∥2 ∼r2 + χ2
d. Now, to ﬁnd conditions for n and d under which with high probability
the neighbors of each point belong to the same cluster, we can utilize the Chi square measure-
concentration bounds [40].
Lemma 3.1 ([40] P.1325). Let X ∼χ2
d. Then
1. P(X −d ≥2√dγ + 2γ) ≤exp(−γ).
2. P(d −X ≥2√dγ) ≤exp(−γ).
Given suﬃciently small γ > 0 we can divide the segment [d, d + r2] to two disjoint segments
of lengths 2√dγ + 2γ and 2√dγ (and solve for d in order to have the total length r2). This
yields
√
d = r2 −2γ
4√γ .
(3)
6

## Page 7

The nearest neighbors of each point will be from the same cluster as long as all distances between
points from the same cluster will be at most d + 2√dγ + 2γ and all distances between points
from diﬀerent clusters will be at least d + r2 −2√dγ. According to lemma 3.1, this will happen
with probability at least (1 −exp(−γ))2n2−n. Denoting this probability as 1 −ϵ and solving for
γ we obtain
γ ≤−log(1 −
(2n2−n)√
1 −ϵ).
(4)
Plugging (4) into (3) we obtain
d = O
 
r4
−log(1 −
(2n2−n)√1 −ϵ)
!
.
(5)
In particular, for ﬁxed n and ϵ, equation (5) implies that the number of nuisance dimensions
must be at most on the order of r4 in order for the clusters to not mix with high probability. In
addition, for a ﬁxed r and ϵ, increasing the number of data points brings the argument inside
the log term arbitrarily close to zero, which implies that for large data, the Laplacian is sensitive
to the number of nuisance dimensions. We support these ﬁndings via experiments, as shown in
Figure 4.
Figure 4: Two cluster datasets. We evaluate the inﬂuence of Gaussian nuisance variables on the
Laplacian. We generate two clusters using 50 samples each with distance r apart in 1-D. We
use d Gaussian nuisance variables and evaluate the leading non trivial eigenvector ψ2 of the
Laplacian. Left: correlation between the second eigenvector ψ1 and the true cluster assignments
y for diﬀerent values of r. As the number of nuisance variables grows, the eigenvector becomes
meaningless. As the distance between cluster grows more nuisance variables are required to
“break” the cluster structure captured by ψ2. Right: by computing the intersection between the
damped correlation curves and 0.7 (shown in the left plot) for diﬀerent values of r we evaluate
the relation between r and number of nuisance variables d required for breaking the cluster
structure. This empirical result supports the analysis presented in 3.2 in which we show that
d = O

r4
−log(1−(2n2−1)√1−ϵ)

. For convenience we added a polynomial ﬁt up to degree 4 presented
as the black line.
7

## Page 8

Figure 5: Evaluating the precision and recall of feature selection, as well as clustering quality
as a function of epoch number. We apply the parameter-free loss variant (see (7)) on the
noisy two-moons data with a total of d features. Left: precision of features selection. Here,
precision is deﬁned as the ratio between amount of retrieved informative features and all retrieved
features, that is
P2
i=1 P(Zi>0)
Pd
i=1 P(Zi>0). Middle: recall of features selection. Here, recall is deﬁned as the
ratio between amount of retrieved informative features and all informative features, that is
P2
i=1 P(Zi>0)
2
. Note that in all of these examples the gates converge to “deterministic” values.
Namely P(Zi > 0) ≃1 for informative features i = 1, 2 and P(Zi > 0) ≃0 for the nuisance
features i = 3, ..., d. Right: clustering accuracy obtained with the retrieved features every 10
epochs. Here, clustering is performed using spectral clustering [25] with a Gaussian kernel.
4
Proposed Method
4.1
Rationale
Recall that the core component of the Laplacian score [27] is the quadratic term f TLf, which
measures the inner product of the feature f with the eigenvectors of the Laplacian L. For
L = Lrw = D−1K a large Laplacian score implies that f has a large component in the subspace
of eigenvectors corresponding to the largest eigenvalues of L. Assuming that the structure of
the data varies slowly, these leading eigenvectors (corresponding to large eigenvalues) manifest
the main structures in the data, hence a large score implies that a feature contributes to the
structure of the data. However, as we demonstrated in the previous section, in the presence of
nuisance features, these leading eigenvectors become less representative of the true structure.
In this regime, one could beneﬁt from evaluations of the Laplacian score when the Laplacian
is computed based on diﬀerent subsets S of features, i.e., of the form f TLXSf, where LXS
is the random walk Laplacian computed based on a subset of features {f ℓ}ℓ∈S. Such gated
Laplacian score would produce a high score for the informative features S∗when the Laplacian
is computed only based on these features, that is when LXS = LXS∗. Searching over all the
diﬀerent feature subsets is infeasible even for a moderate number of features. Fortunately, we
can use continuous stochastic “gating” functions to explore the space of feature subsets.
Speciﬁcally, we propose to apply diﬀerential stochastic gates to the input features, and
compute the Laplacian score after multiplying the input features with the gates.
Taking
advantage of the fact that informative features are expected to have higher scores than nuisance
ones, we penalize open gates. By applying gradient decent to a cost function based on LXS we
obtain the desired dynamic, in which gates corresponding to features that contain high level of
noise will gradually close, while gates corresponding to features that are consistent with the
true structures in the data will gradually get fully open. This is demonstrated in Fig. 3.
8

## Page 9

4.2
Diﬀerentiable Unsupervised Feature Selection (DUFS)
Let X ∈Rm×d be a data minibatch. Let Z ∈[0, 1]d be a random variable representing the
stochastic gates, parametrized by µ ∈Rd, as deﬁned in Section 2. For each mini-batch we draw
a vector z of realizations from Z and deﬁne a matrix ˜Z ∈[0, 1]m×d consisting of m copies of z.
We denote ˜X
△= X ⊙˜Z as gated input, where ⊙is an element-wise multiplication, also known
as Hadamard product. Let L ˜
X be the random walk graph Laplacian computed on ˜X.
We propose two loss function variants.
Both variants contain a feature scoring term
−1
mTr[ ˜X
TL ˜
X ˜X], and a feature selection regularization term Pd
i=1 P(Zi ≥0), following (2). In
the ﬁrst variant (6) the two terms are balanced using a hyperparameter λ ≥0.
L(µ; λ) := −
Tr
h ˜X
TL ˜
X ˜X
i
m
+ λ
d
X
i=1
P(Zi ≥0).
(6)
Controlling λ allows for ﬂexibility in the number of selected features. To obviate the need to
tune λ, we propose a second loss function, which is parameter-free
Lparam-free(µ) := −
Tr
h ˜X
TL ˜
X ˜X
i
m Pd
i=1 P(Zi ≥0) + δ ,
(7)
where δ is a small constant added to circumvent division by 0. The parameter-free variant (7)
seeks to minimize the average score per selected feature, where the average is calculated as
the total score (in the numerator) divided by a proxy for the number of selected features (the
denominator). Minimizing both proposed objectives (6) and (7) will encourage the gates to
remain open for features that yield high Laplacian score, and closed for the remaining features.
Our algorithm involves applying a standard optimization scheme (such as stochastic gradient
decent) to objective (6) or (7). After training, we remove the stochasticity (ϵi in (1)) from the
gates and retain features such that Zi > 0.
4.2.1
Raising L to the t’th Power
Replacing the Laplacian L in equations (6) and (7) by its t-th power Lt with t > 1 corresponds
to taking t random walk steps [39]. This suppresses the smallest eigenvalues of the Laplacian,
while preserving its eigenvectors. We used t = 2, which was observed to improve the performance
of our proposed approach (see Appendix for more details).
5
Experiments
To demonstrate the capabilities of DUFS, we begin by presenting an artiﬁcial “two-moons”
experiment. We then report results obtained on several standard datasets, and compare them
to current baselines. When applying the method to real data we perform feature selection based
on Eq. (6) using several values of λ. Next, followin the analysis in [41], we perform clustering
using k-means based on the leading 50, 100, 150, 200, 250, or 300 selected features and average
the results over 20 runs. Leading features are identiﬁed by sorting the gates based on P(Zi) (see
(2)). The number k of clusters is set as the number of classes and labels are utilized to evaluate
clustering accuracy. The best average clustering accuracy is recorded along with the number |S|
of selected features.
9

## Page 10

Figure 6: Noisy image experiments. Top: examples of noisy MNIST digits highlighted with the
leading 50 features selected by DUFS (left) and LS (right). Bottom: examples from the noisy
PIX10 datasets overlaid with the leading 300 features selected by DUFS (left) and LS (right).
This ﬁgure is best viewed in color. The gray scale of MNIST images is inverted to improve
visibility.
5.1
Noisy Two Moons
In this experiment, we use a two-moons shaped dataset (see Fig. 2) concatenated with nuisance
features. The ﬁrst two coordinates f 1, f 2 are generated by adding a Gaussian noise with zero
mean and variance of σ2
r = 0.1 onto two nested half circles. Nuisance features f i, i = 3, ..., d,
are drawn from a multivariate Gaussian distribution with zero mean and identity covariance.
The total number of samples is n = 100. Note that the small sample size makes the task of
identifying nuisance variables more challenging.
We evaluate the convergence of the parameter-free loss (7) using gradient decent. We use
diﬀerent number of features d and plot the precision and recall of feature selection throughout
training (see Fig. 5). In all of the presented examples, perfect precision and recall are achieved
at convergence.
Figure 7: Clustering accuracy on three real world datasets. Clustering was performed by
applying k-means to features selected by DUFS and LS. The averages and standard deviations
based on 20 runs are shown.
5.2
Noisy Image Data
In the following experiment we evaluate our method on two noisy image datasets. The ﬁrst is a
noisy variant of MNIST [42], in which each background pixel is replaced by a random value
drawn uniformly from [0, 1] (see also [43]). Here we focus only on the digits ‘3‘ and ‘8‘. The
second dataset is a noisy variant of PIXRAW10P (abbreviated PIX10), created by adding noise
drawn uniformly from [0, 0.3] to all pixels. In both datasets the images were scaled into [0, 1]
prior to the addition of noise. We applied to both datasets the DUFS and LS approaches. In
10

## Page 11

the top panels of Fig. 6 we present the leading 50 features retained on noisy MNIST along
with the average clustering accuracy over 20 runs of k-means. In this case, DUFS’ open gates
concentrate at the left side of the handwriting area, which is the side that distinguishes ‘3‘ from
‘8‘. This allows DUFS to achieve a higher clustering accuracy comparing to LS . We remark
that the training was purely unsupervised and all label information was absent. The bottom
panels of Fig. 6 show the leading 300 features retained on noisy PIX10 along with the average
clustering accuracy. Here, DUFS selects features which are more informative for clustering the
face images. We refer the reader to Appendix S2 for additional information on the noisy image
datasets along with extended results on noisy images.
5.3
Clustering of Real World Data
Here, we evaluate the capabilities of the proposed approach (DUFS) in clustering of real datasets.
We use several benchmark datasets12. The properties of all datasets are summarized in Table 1.
We compare DUFS to Laplacian Score [26] (LS), Multi-Cluster Feature Selection [44] (MCFS),
Local Learning based Clustering (LLCFS) [45], Nonnegative Discriminative Feature Selection
(NDFS) [46], Multi-Subspace Randomization and Collaboration (SRCFS) [47] and Concrete
Auto-encoders (CAE) [19]. In table 1 we present the accuracy of clustering based on feature
selected by DUFS, the 6 baselines and based on all features (All). As can be seen, DUFS
outperforms all baselines on 6 datasets, and is ranked at second place on the remaining 3.
Overall the median and mean ranking of DUFS are 1 and 1.3.
In the next experiment we evaluate the eﬀectiveness of the proposed method for diﬀerent
numbers of selected features on 3 datasets. We compare DUFS versus LS by performing k-means
clustering using the features selected by each method. In Fig. 7 we present the clustering
accuracies (averaged over 20 runs) based on the leading {50, 100, ..., 300} features. We see that
DUFS consistently selects features which provide higher clustering capabilities compared to LS.
1http://featureselection.asu.edu/datasets.php
2For RCV1 we use a binary subset (see exact details in the Appendix) of the full data: https://scikit-
learn.org/0.18/datasets/rcv1.html
Datasets
LS
MCFS
NDFS
LLCFS
SRCFS
CAE
DUFS
All
Dim/Samples/Classes
GISETTE
75.8 (50)
56.5 (50)
69.3 (250)
72.5 (50)
68.5 (50)
77.3 (250)
99.5 (50)
74.4
4955 / 6000 / 2
PIX10
76.6 (150)
75.9 (200)
76.7 (200)
69.1 (300)
76.0 (300)
94.1 (250)
88.4 (50)
74.3
10000 / 100 / 10
COIL20
55.2 (250)
59.7 (250)
60.1 (300)
48.1 (300)
59.9 (300)
65.6 (200)
65.8 (250)
53.6
1024 / 1444 / 20
Yale
42.7 (300)
41.7 (300)
42.5 (300)
42.6 (300)
46.3 (250)
45.4 (250)
47.9 (200)
38.3
1024 / 165 / 15
TOX-171
47.5 (200)
42.5 (100)
46.1 (100)
46.7 (250)
45.8 (150)
44.4 (150)
49.1 (50)
41.5
2000 / 62 / 2
ALLAML
73.2 (150)
68.4 (100)
69.4 (100)
77.8 (50)
67.7 (250)
72.2 (200)
74.5 (100)
67.3
7192 / 72 / 2
PROSTATE
57.5 (300)
57.3 (300)
58.3 (100)
57.8 (50)
60.6 (50)
56.9 (250)
64.7 (150)
58.1
5966 / 102 / 2
RCV1
54.9 (300)
50.1 (150)
55.1 (150)
55.0 (300)
53.7 (300)
54.9 (300)
60.2 (300)
50.0
47236 / 21232 / 2
ISOLET
56.4 (300)
62.1 (300)
60.4 (100)
52.3 (100)
60.1 (250)
63.8 (250)
63.6 (150)
60.8
617 / 1560 / 26
Median rank
4
6
4
4
5
3
1
Mean rank
4.1
6
3.9
4.6
4.7
3.4
1.3
Table 1: Left table- Average clustering accuracy on several benchmark datasets. Clustering is
performed by applying k-means to the features selected by the diﬀerent methods. The number
of selected features is shown in parenthesis. Right table- Properties of the real world data used
for empirical evaluation.
11

## Page 12

6
Conclusions
In this paper, we propose DUFS, a novel unsupervised feature selection method by introducing
learnable Bernoulli gates into a Laplacian score. DUFS has an advantage over the standard
Laplacian score as it re-evaluates the Laplacian score based on the subset of selected features. We
demonstrate that our proposed approach captures structures in the data that are not detected
by mining the data with standard Laplacian, in the presence of nuisance features. Finally, we
experimentally demonstrate that our method outperforms current unsupervised feature selection
baselines on several real-world datasets.
Acknowledgements
The authors thank Stefan Steinerberger, Boaz Nadler and Ronen Basri for helpful discussions.
This work was supported by the National Institutes of Health [R01GM131642, UM1DA051410,
P50CA121974 and R61DA047037].
References
[1] R. Battiti, “Using mutual information for selecting features in supervised neural net
learning,” IEEE Transactions on neural networks, vol. 5, no. 4, pp. 537–550, 1994.
[2] H. Peng, F. Long, and C. Ding, “Feature selection based on mutual information criteria
of max-dependency, max-relevance, and min-redundancy,” IEEE Transactions on pattern
analysis and machine intelligence, vol. 27, no. 8, pp. 1226–1238, 2005.
[3] P. A. Estévez, M. Tesmer, C. A. Perez, and J. M. Zurada, “Normalized mutual information
feature selection,” IEEE Transactions on Neural Networks, vol. 20, no. 2, pp. 189–201,
2009.
[4] L. Song, A. Smola, A. Gretton, K. M. Borgwardt, and J. Bedo, “Supervised feature selection
via dependence estimation,” in Proceedings of the 24th international conference on Machine
learning, pp. 823–830, ACM, 2007.
[5] L. Song, A. Smola, A. Gretton, J. Bedo, and K. Borgwardt, “Feature selection via depen-
dence maximization,” Journal of Machine Learning Research, vol. 13, no. May, pp. 1393–
1434, 2012.
[6] J. Chen, M. Stern, M. J. Wainwright, and M. I. Jordan, “Kernel feature selection via con-
ditional covariance minimization,” in Advances in Neural Information Processing Systems,
pp. 6946–6955, 2017.
[7] R. Kohavi and G. H. John, “Wrappers for feature subset selection,” Artiﬁcial intelligence,
vol. 97, no. 1-2, pp. 273–324, 1997.
[8] G. Stein, B. Chen, A. S. Wu, and K. A. Hua, “Decision tree classiﬁer for network intrusion
detection with ga-based feature selection,” in Proceedings of the 43rd annual Southeast
regional conference-Volume 2, pp. 136–141, ACM, 2005.
[9] Z. Zhu, Y.-S. Ong, and M. Dash, “Wrapper–ﬁlter feature selection algorithm using a
memetic framework,” IEEE Transactions on Systems, Man, and Cybernetics, Part B
(Cybernetics), vol. 37, no. 1, pp. 70–76, 2007.
12

## Page 13

[10] J. Reunanen, “Overﬁtting in making comparisons between variable selection methods,”
Journal of Machine Learning Research, vol. 3, no. Mar, pp. 1371–1382, 2003.
[11] G. I. Allen, “Automatic feature selection via weighted kernels and regularization,” Journal
of Computational and Graphical Statistics, vol. 22, no. 2, pp. 284–299, 2013.
[12] R. Tibshirani, “Regression shrinkage and selection via the lasso,” Journal of the Royal
Statistical Society. Series B (Methodological), pp. 267–288, 1996.
[13] Y. Yamada, O. Lindenbaum, S. Negahban, and Y. Kluger, “Feature selection using stochastic
gates,” arXiv preprint arXiv:1810.04247, 2018.
[14] Y. Li, C.-Y. Chen, and W. W. Wasserman, “Deep feature selection: theory and application
to identify enhancers and promoters,” Journal of Computational Biology, vol. 23, no. 5,
pp. 322–336, 2016.
[15] S. Scardapane, D. Comminiello, A. Hussain, and A. Uncini, “Group sparse regularization
for deep neural networks,” Neurocomput., vol. 241, pp. 81–89, June 2017.
[16] J. Feng and N. Simon, “Sparse-Input Neural Networks for High-dimensional Nonparametric
Regression and Classiﬁcation,” ArXiv e-prints, Nov. 2017.
[17] S. Wang, Z. Ding, and Y. Fu, “Feature selection guided auto-encoder,” in Thirty-First
AAAI Conference on Artiﬁcial Intelligence, 2017.
[18] K. Han, Y. Wang, C. Zhang, C. Li, and C. Xu, “Autoencoder inspired unsupervised
feature selection,” in 2018 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), pp. 2941–2945, IEEE, 2018.
[19] A. Abid, M. F. Balin, and J. Zou, “Concrete autoencoders for diﬀerentiable feature selection
and reconstruction,” arXiv preprint arXiv:1901.09346, 2019.
[20] O. Alter, P. O. Brown, and D. Botstein, “Singular value decomposition for genome-wide
expression data processing and modeling,” Proceedings of the National Academy of Sciences,
vol. 97, no. 18, pp. 10101–10106, 2000.
[21] R. Varshavsky, A. Gottlieb, M. Linial, and D. Horn, “Novel unsupervised feature ﬁltering
of biological data,” Bioinformatics, vol. 22, no. 14, pp. e507–e513, 2006.
[22] D. Devakumari and K. Thangavel, “Unsupervised adaptive ﬂoating search feature selection
based on contribution entropy,” in 2010 international conference on communication and
computational intelligence (INCOCCI), pp. 623–627, IEEE, 2010.
[23] M. Banerjee and N. R. Pal, “Feature selection with svd entropy: Some modiﬁcation and
extension,” Information Sciences, vol. 264, pp. 118–134, 2014.
[24] V. M. Rao and V. Sastry, “Unsupervised feature ranking based on representation entropy,” in
2012 1st International Conference on Recent Advances in Information Technology (RAIT),
pp. 421–425, IEEE, 2012.
[25] A. Y. Ng, M. I. Jordan, and Y. Weiss, “On spectral clustering: Analysis and an algorithm,”
in Advances in neural information processing systems, pp. 849–856, 2002.
13

## Page 14

[26] M. Belkin and P. Niyogi, “Laplacian eigenmaps and spectral techniques for embedding and
clustering,” in Advances in neural information processing systems, pp. 585–591, 2002.
[27] X. He, D. Cai, and P. Niyogi, “Laplacian score for feature selection,” in Advances in neural
information processing systems, pp. 507–514, 2006.
[28] Z. Zhao and H. Liu, “Spectral feature selection for supervised and unsupervised learning,”
in Proceedings of the 24th international conference on Machine learning, pp. 1151–1157,
2007.
[29] P. Padungweang, C. Lursinsap, and K. Sunat, “Univariate ﬁlter technique for unsupervised
feature selection using a new laplacian score based local nearest neighbors,” in 2009
Asia-Paciﬁc Conference on Information Processing, vol. 2, pp. 196–200, IEEE, 2009.
[30] Z. A. Zhao and H. Liu, Spectral Feature Selection for Data Mining (Open Access). CRC
Press, 2011.
[31] U. Von Luxburg, “A tutorial on spectral clustering,” Statistics and computing, vol. 17,
no. 4, pp. 395–416, 2007.
[32] A. Miller, N. Foti, A. D’Amour, and R. P. Adams, “Reducing reparameterization gradient
variance,” in Advances in Neural Information Processing Systems, pp. 3708–3718, 2017.
[33] M. Figurnov, S. Mohamed, and A. Mnih, “Implicit reparameterization gradients,” in
Advances in Neural Information Processing Systems, pp. 441–452, 2018.
[34] X. He and P. Niyogi, “Locality preserving projections,” in Advances in neural information
processing systems, pp. 153–160, 2004.
[35] C. J. Maddison, A. Mnih, and Y. W. Teh, “The concrete distribution: A continuous
relaxation of discrete random variables,” arXiv preprint arXiv:1611.00712, 2016.
[36] E. Jang, S. Gu, and B. Poole, “Categorical reparameterization with gumbel-softmax,” 2017.
[37] C. Louizos, M. Welling, and D. P. Kingma, “Learning sparse neural networks through l_0
regularization,” arXiv preprint arXiv:1712.01312, 2017.
[38] E. Jang, S. Gu, and B. Poole, “Categorical reparameterization with gumbel-softmax,” arXiv
preprint arXiv:1611.01144, 2016.
[39] B. Nadler, S. Lafon, R. Coifman, and I. G. Kevrekidis, “Diﬀusion maps-a probabilistic
interpretation for spectral embedding and clustering algorithms,” in Principal manifolds
for data visualization and dimension reduction, pp. 238–260, Springer, 2008.
[40] B. Laurent and P. Massart, “Adaptive estimation of a quadratic functional by model
selection,” Annals of Statistics, pp. 1302–1338, 2000.
[41] S. Wang, J. Tang, and H. Liu, “Embedded unsupervised feature selection,” in Twenty-ninth
AAAI conference on artiﬁcial intelligence, 2015.
[42] Y. LeCun, L. Bottou, Y. Bengio, and P. Haﬀner, “Gradient-based learning applied to
document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
14

## Page 15

[43] Y. Qi, H. Wang, R. Liu, B. Wu, Y. Wang, and G. Pan, “Activity-dependent neuron model
for noise resistance,” Neurocomputing, vol. 357, pp. 240–247, 2019.
[44] D. Cai, C. Zhang, and X. He, “Unsupervised feature selection for multi-cluster data,” in
Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery
and data mining, pp. 333–342, 2010.
[45] H. Zeng and Y.-m. Cheung, “Feature selection and kernel learning for local learning-based
clustering,” IEEE transactions on pattern analysis and machine intelligence, vol. 33, no. 8,
pp. 1532–1547, 2010.
[46] Z. Li, Y. Yang, J. Liu, X. Zhou, and H. Lu, “Unsupervised feature selection using nonnegative
spectral analysis,” in Twenty-Sixth AAAI Conference on Artiﬁcial Intelligence, 2012.
[47] D. Huang, X. Cai, and C.-D. Wang, “Unsupervised feature selection with multi-subspace
randomization and collaboration,” Knowledge-Based Systems, vol. 182, p. 104856, 2019.
[48] A. Singer, R. Erban, I. G. Kevrekidis, and R. R. Coifman, “Detecting intrinsic slow variables
in stochastic dynamical systems by anisotropic diﬀusion maps,” Proceedings of the National
Academy of Sciences, vol. 106, no. 38, pp. 16090–16095, 2009.
[49] Y. Keller, R. R. Coifman, S. Lafon, and S. W. Zucker, “Audio-visual group recognition
using diﬀusion maps,” IEEE Transactions on Signal Processing, vol. 58, no. 1, pp. 403–413,
2009.
[50] L. Zelnik-Manor and P. Perona, “Self-tuning spectral clustering,” in Advances in neural
information processing systems, pp. 1601–1608, 2005.
[51] O. Lindenbaum, M. Salhov, A. Yeredor, and A. Averbuch, “Kernel scaling for manifold
learning and classiﬁcation,” arXiv preprint arXiv:1707.01093, 2017.
15

## Page 16

Appendix
S1
Tuning the Kernel’s Bandwidth
It is important to properly tune the kernel scale/bandwidth σb, which determines the scale
of connectivity of the kernel K. Several studies have proposed schemes for tuning σb, see
for example [48, 49, 50, 51]. Here, we focus on two schemes, a global bandwidth and a local
bandwidth. The local bandwidth proposed in [50], involves setting a local-scale σi for each data
point xi, i = 1, ..., n. The scale is chosen using the L1 distance from the k-th nearest neighbor
of the point xi. Explicitly, the calculation for each point is
σi = C · ||xi −xk||2, i = 1, ..., N,
(8)
where xk is the k-th nearest (Euclidean) neighbor of the point xi, and C is a predeﬁned constant
in the range [1, 5]. We compute ˆσb as the max over σi, then, the value of the kernel for points
xi and xj is
Ki,j = exp
 
−||xi −xj||2
ˆσb
!
, i, j ∈{1 . . . n}.
(9)
This scale guarantees that all of the points are connected to at least k neighbors.
S2
Feature selection on Image Datasets
Here, we provide a deeper look into the features identiﬁed by the proposed method when applied
to image data. We start with COIL20 which is a data that contains 20 objects captured at
diﬀerent viewing angles. In Fig. S1 we present the leading {50, 100, ..., 300} features selected by
DUFS and LS along with the average clustering accuaracies based on the selected features. In
this example DUFS selects features which lie on the symmetry axis of COIL20, these features
are more informative for clustering COIL20 since the values of rotated objects vary slowly on
this axis. Next, we present a similar comparison on COIL100. COIL100 contains 7200 samples
of 100 objects captured at diﬀerent angles. Each image is of dimension [128, 128, 3]. In Fig.
S2 we present the leading {50, 100, ..., 300} features selected by DUFS and LS along with the
average clustering accuracies based on the selected features. Here, feature selection is performed
based on a black and white version of the RGB image and clustering is performed based on the
corresponding subset of pixels from the RGB tensor.
Finally, in Fig. S3 we present the results of application of DUFS to the noisy MNIST dataset.
This is an extension of the results presented in the paper. Speciﬁcally, we demonstrate the
clustering accuracies based on the leading {50, 100, ..., 300} features selected by DUFS and LS.
In this experiment, we focused on a random subset of 1000 samples of the digits 3 and 8.
S3
Raising L to the t’th Power
To suppress the smallest eigenvalues of the Laplacian, we have suggested to replace the Laplacian
L in equations (6) and (7) by its t-th power Lt with t > 1. As shown in [39] this corresponds
for taking t random walk steps on the graph of the data. In this subsection we empirically
demonstrate the eﬀect of t using the two-moons dataset (described in the Experimental section
of the main text). We construct the two-moons dataset with diﬀerent number of nuisance
variables (d) and apply DUFS (with the parameter free loss) computed based on L raised to
16

## Page 17

the power of t = 1, 2 and 3. In Fig. S4 we present the clustering accuracy (averaged over 100
runs) based on k-means, which is applied to the selected features. As evident in this plot the
Laplacian based on t = 2 yields better performance for a wider range of nuisance variables in
this experiment. Following this result, we keep t = 2 in all of our examples.
S4
Experimental Details
In this subsection we describe all required details for conducting the experiments provided in
the main text. We use SGD for all the experiments which are conducted using Intel(R) Xeon(R)
CPU E5-2620 v3 @2.4Ghz x2 (12 cores total). For LS, MCFS, and NDFS we use a python
implementation from 3. For LLCFS and SRCFS we use a Matlab implementation from 4 and
5. For CAE we use a python implementation available at 6. For DUFS and LS, we use k = 2
(number of nearest neighbors) which worked well on all of the datastes, except for ISOLET
and GISSETE datasets in which we used k = 5. The factor C (see Eq. 8) in all experiments
is 5 except for COIL20 and PIX10 in which C = 2. All datasets are publicly available at 7,
except RCV1 which is available at 8. RCV1 is a multi-class multi-label datasets, in our analysis
we use a binary subset of RCV1. To create this subset, we focus on the ﬁrst two classes and
remove all samples that have multiple labels, then we balance the classes by down sampling
the larger class. For NDFS, MCFS and SRCFS we use k = 5 for the aﬃnity matrix W , note
that NDFS and LLCFS use the number of clusters for selecting features. The tuning process for
hyper-parameters of all method follows the grid search described described in [41].
In all examples except RCV1, COIL100 and COIL20 we use a full batch size for computing
the kernel, for COIL100 and COIL20 the batch size is 1000. For all two-moons examples
presented in Fig. 2 we use the parameter free loss term with a learning rate (LR) of 1 and 5000
epochs. For PROSTATE data, we use a learning rate of 1, 12000 epochs and λ is evaluated
in the range [0.01, 1]. For GLIOMA data we use a learning rate of 0.3, 12000 epochs and λ is
evaluated in the range [3, 30]. For ALLAML data we use a learning rate of 0.3, 20000 epochs
and λ is evaluated in the range [1, 5]. For COIL20 data we use a learning rate of 0.3, 26000
epochs and λ is evaluated in the range [0.01, 2]. For COIL100 data we use a learning rate of 1,
6000 epochs and λ is evaluated in the range [0.01, 2]. For PIX10 data we use a learning rate of
0.3, 20000 epochs and λ is evaluated in the range [0.05, 1]. For ISOLET data we use a learning
rate of 0.3, 20000 epochs and λ is evaluated in the range [0.1, 10].
3https://github.com/jundongl/scikit-feature
4https://github.com/huangdonghere/SRCFS
5https://www.mathworks.com/matlabcentral/ﬁleexchange/56937-feature-selection-library
6https://github.com/mfbalin/Concrete-Autoencoders
7http://featureselection.asu.edu/datasets.php
8https://scikit-learn.org/0.18/datasets/rcv1.html
17

## Page 18

(a) DUFS
(b) LS
Figure S1: Features selected by DUFS and LS in the COIL20 dataset. Top: selected features
(cyan dots) and clustering accuracy based on DUFS. Note that as COIL20 contains diﬀerent
angles of each object, the selected feature lie approximately on the symmetry axis. Bottom:
selected features (magenta dots) and clustering accuracy based on LS.
18

## Page 19

(a) DUFS
(b) LS
Figure S2: Same as for S1 but for the COIL100 dataset. Note that as COIL100 contains diﬀerent
angles of each object, the selected feature lie approximately on the symmetry axis. In this
example, the LS also selects features on the symmetry axis, however the LS based selected
features are condensed at a small region near the top part of the image. These features are
informative for clustering wide vs. long objects but less informative for clustering all 100 objects.
19

## Page 20

(a) DUFS
(b) LS
Figure S3: Selected features on MNIST dataset. Top: selected features and clustering accuracy
based on DUFS. Bottom: selected features and clustering accuracy based on LS. In this example,
DUFS outperforms the LS when it is regularized to select a small number of features. However,
when the regularization is set to select > 100 features in DUFS, the features with top scores in
both methods are similar.
20

## Page 21

Figure S4: Taking powers of the graph Laplacian. Clustering accuracy vs. number of nuisance
dimensions in the two-moons datasts. We apply the parameter free variant of DUFS using a
Laplacian L raised to the power of t. Clustering is performed using k-means applied to the
selected features and averaged over 100 runs of DUFS.
21
