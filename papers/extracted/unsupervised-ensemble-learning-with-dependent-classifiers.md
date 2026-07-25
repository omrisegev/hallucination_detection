---
source_pdf: papers/Unsupervised Ensemble Learning with Dependent Classifiers.pdf
slug: unsupervised-ensemble-learning-with-dependent-classifiers
pages: 21
extracted_on: 2026-07-13
---

# Unsupervised Ensemble Learning with Dependent Classifiers

## Page 1

arXiv:1510.05830v2  [cs.LG]  23 Feb 2016
Unsupervised Ensemble Learning with Dependent Classiﬁers
Ariel Jaﬀe1∗, Ethan Fetaya1, Boaz Nadler1, Tingting Jiang2 and Yuval Kluger2,3
1Dept. of Computer Science and Applied Mathematics, Weizmann Institute of Science, Rehovot Israel 76100
2Interdepartmental Program in Computational Biology and Bioinformatics, Yale University, New Haven, CT 06511
3Dept. of Pathology and Yale Cancer Center, Yale University School of Medicine, New Haven, CT 06520, USA
Abstract
In unsupervised ensemble learning, one obtains predictions from multiple sources or classi-
ﬁers, yet without knowing the reliability and expertise of each source, and with no labeled data
to assess it. The task is to combine these possibly conﬂicting predictions into an accurate meta-
learner. Most works to date assumed perfect diversity between the diﬀerent sources, a property
known as conditional independence. In realistic scenarios, however, this assumption is often
violated, and ensemble learners based on it can be severely sub-optimal. The key challenges
we address in this paper are: (i) how to detect, in an unsupervised manner, strong violations of
conditional independence; and (ii) construct a suitable meta-learner. To this end we introduce
a statistical model that allows for dependencies between classiﬁers. Our main contributions are
the development of novel unsupervised methods to detect strongly dependent classiﬁers, better
estimate their accuracies, and construct an improved meta-learner. Using both artiﬁcial and
real datasets, we showcase the importance of taking classiﬁer dependencies into account and the
competitive performance of our approach.
1
Introduction
In recent years unsupervised ensemble learning has become increasingly popular. In multiple appli-
cation domains one obtains the predictions, over a large set of unlabeled instances, of an ensemble of
diﬀerent experts or classiﬁers with unknown reliability. Common tasks are to combine these possibly
conﬂicting predictions into an accurate meta-learner, as well as assessing the accuracy of the various
experts, both without any labeled data.
A leading example is crowdsourcing, whereby a tedious labeling task is distributed to many
annotators. Unsupervised ensemble learning is of increasing interest also in computational biology,
where recent works in the ﬁeld propose to solve diﬃcult prediction tasks by applying multiple
algorithms and merging their results [1, 3, 7, 14].
Additional examples of unsupervised ensemble
learning appear, among others, in medicine [12] and decision science [17].
Perhaps the ﬁrst to address ensemble learning in this fully unsupervised setup were Dawid and
Skene [5]. A key assumption in their work was of perfect diversity between the diﬀerent classiﬁers.
Namely, their labeling errors were assumed statistically independent of each other. This property,
known as conditional independence is illustrated in the graphical model of Fig. 1 (left). In [5],
Dawid and Skene proposed to estimate the parameters of the model, i.e.
the accuracies of the
diﬀerent classiﬁers, by the EM procedure on the non-convex likelihood function. With the increasing
popularity of crowdsourcing and other unsupervised ensemble learning applications, there has been
∗Email addresses: Ariel Jaﬀe: ariel.jaffe@weizmann.ac.il, Ethan Fetaya: ethan.fetaya@weizmann.ac.il,Boaz
Nadler:
boaz.nadler@weizmann.ac.il,Tingting
Jiang:
tingting.jiang@yale.edu,Yuval
Kluger:
yuval.kluger@yale.edu
1

## Page 2

a surge of interest in this line of work, and multiple extensions of it [11,18,20,22,23]. As the quality
of the solution found by the EM algorithm critically depends on its starting point, several recent
works derived computationally eﬃcient spectral methods to suggest a good initial guess [2,9,10,15].
Despite its popularity and usefulness, the model of Dawid and Skene has several limitations.
One notable limitation is its assumption that all instances are equally diﬃcult, with each classiﬁer
having the same probability of error over all instances. This issue was addressed, for example, by
Whitehill et. al. [23] who introduced a model of instance diﬃculty, and also by Tian et. al. [21]
who proposed a model where instances are divided into groups, and the expertise of each classiﬁer
is group dependent.
A second limitation, at the focus of our work, is the assumption of perfect conditional inde-
pendence between all classiﬁers. As we illustrate below, this assumption may be strongly violated
in real-world scenarios. Furthermore, as shown in Sec. 5, neglecting classiﬁer dependencies may
yield quite sub-optimal predictions. Yet, to the best of our knowledge, relatively few works have
attempted to address this important issue.
To handle classiﬁer dependencies, Donmez et. al. [6] proposed a model with pairwise interactions
between all classiﬁer outputs. However, they noted that empirically, their model did not yield more
accurate predictions. Platanios et. al. [16] developed a method to estimate the error rates of either
dependent or independent classiﬁers.
Their method is based on analyzing the agreement rates
between pairs or larger subsets of classiﬁers, together with a soft prior on weak dependence amongst
them.
The present work is partly motivated by the ongoing somatic mutation DREAM (Dialogue for
Reverse Engineering Assessments and Methods) challenge, a sequence of open competitions for
detecting irregularities in the DNA string. This is a real-world example of unsupervised ensemble
learning, where participants in the currently open competition are given access to the predictions
of more than 100 diﬀerent classiﬁers, over more than 100,000 instances.
These classiﬁers were
constructed by various labs worldwide, each employing their own biological knowledge and possibly
proprietary labeled data. The task is to construct, in an unsupervised fashion, an accurate ensemble
learner.
In ﬁgure 2a we present the empirical conditional covariance matrix between diﬀerent classiﬁers
in one of the databases of the DREAM challenge, for which ground truth labels have been disclosed.
Under the conditional independence assumption, the population conditional covariance between
every two classiﬁers should be exactly zero. Figure 2a, in contrast, exhibits strong dependencies
between groups of classiﬁers.
Unsupervised ensemble learning in the presence of possibly strongly dependent classiﬁers raises
the following two key challenges: (i) detect, in an unsupervised manner, strong violations of condi-
tional independence; and (ii) construct a suitable meta-learner.
To cope with these challenges, in Sec. 2 we introduce a new model for the joint distribution of
all classiﬁers which allows for dependencies between them through an intermediate layer of latent
variables. This generalizes the model of Dawid and Skene, and allows for groups of strongly correlated
classiﬁers, as observed for example in the DREAM data.
In Sec. 3 we devise a simple algorithm to detect subsets of strongly dependent classiﬁers using
only their predictions and no labeled data.
This is done by exploiting the structural low-rank
properties of the classiﬁers’ covariance matrix. Figure 2b shows our resulting estimate for deviations
from conditional independence on the same data as ﬁgure 2a. Comparing the two ﬁgures illustrates
the ability of our method to detect strong dependencies with no labeled data.
In Sec. 4 we propose methods to better estimate the accuracies of the classiﬁers and construct an
improved meta-learner, both in the presence of strong dependencies between some of the classiﬁers.
Finally, in Sec. 5 we illustrate the competitive performance of the modiﬁed ensemble-learner derived
from our model on both artiﬁcial data, four datasets from the UCI repository and three datasets from
the DREAM challenge. These empirical results showcase the limitations of the strict conditional
independence model, and highlight the importance of modeling the statistical dependencies between
2

## Page 3

Y
f1
fi
fm
ψ1, η1
ψi, ηi
ψm, ηm
Y
α1
αk
αK
f1
f2
fi
fi+1
fm
ψα
1 , ηα
1
ψα
i , ηα
i
ψα
m, ηα
m
Fig. 1: (Left) The perfect conditional independence model of Dawid and Skene. All classiﬁers are
independent given the class label Y ; (Right) The generalized model considered in this work.
diﬀerent classiﬁers in unsupervised ensemble learning scenarios.
2
Problem Setup
Notations.
We consider the following binary classiﬁcation problem. Let X be an instance space
with an output space Y = {−1, 1}. A labeled instance (x, y) ∈X × Y is a realization of the random
variable (X, Y ). The joint distribution p(x, y), as well as the marginals pX(x) and pY (y), are all
unknown. We further denote by b the class imbalance of Y ,
b = pY (1) −pY (−1).
(1)
Let {fi}m
i=1 be a set of m binary classiﬁers operating on X. As our classiﬁcation problem is binary,
the accuracy of the i-th classiﬁer is fully characterized by its sensitivity ψi and speciﬁcity ηi,
ψi = Pr (fi(X) = 1|Y = 1)
ηi = Pr (fi(X) = −1|Y = −1) .
(2)
For future use, we denote by πi its balanced accuracy, given by the average of its sensitivity and
speciﬁcity
πi = 1
2(ψi + ηi).
(3)
Note that when the class imbalance is zero, πi is simply the overall accuracy of the i-th classiﬁer.
The classical conditional independence model.
In the model proposed by Dawid and Skene
[5], depicted in Fig. 1(left), all m classiﬁers were assumed conditionally independent given the class
label. Namely, for any set of predictions a1, . . . , am ∈{±1}
Pr(f1 = a1, . . . , fm = am|Y ) =
Y
i
Pr(fi = ai|Y ).
(4)
As shown in [5], the maximum likelihood estimation (MLE) for y given the parameters ψi, ηi and b
is linear in the predictions of f1, ..., fm
ˆy = sign
 m
X
i=1
wifi(x) + w0
!
, wi = w(ψi, ηi).
(5)
Hence, the main challenge is to estimate the model parameters ψi and ηi. A simple approach to
do so, as described in [9,15], is based on the following insight: A classiﬁer which is totally random
3

## Page 4

has zero correlation with any other classiﬁer. In contrast, a high correlation between the predictions
of two classiﬁers is a strong indication that both are highly accurate, assuming they are not both
adversarial.
In many realistic scenarios, however, an ensemble may contain several strongly dependent clas-
siﬁers.
Such a scenario has several consequences: First, the above insight that high correlation
between two classiﬁers implies that both are accurate breaks down completely. Second, as shown
in Sec. 5, estimating the classiﬁers parameters ψi, ηi as if they were conditionally independent may
be highly inaccurate. Third, in contrast to Eq. (5), the optimal ensemble learner is in general
non-linear in the m classiﬁers. Applying the linear meta-classiﬁer of Eq. (5) may be suboptimal,
even when provided with the true classiﬁer accuracies.
A model for conditionally dependent classiﬁers.
In this paper we signiﬁcantly relax the
conditional independence assumption.
We introduce a new model which allows classiﬁers to be
dependent through unobserved latent variables, and develop novel methods to learn the model
parameters and construct an improved non-linear meta-learner.
In contrast to the 2-layer model of Dawid and Skene, our proposed model, illustrated in Fig.
1(right), has an additional intermediate layer with K ≤m latent binary random variables {αk}K
k=1.
In this model, the unobserved αk are conditionally independent given the true label Y , whereas each
observed classiﬁer depends on Y only through a single and unknown latent variable. Classiﬁers that
depend on diﬀerent latent variables are thus conditionally independent given Y , whereas classiﬁers
that depend on the same latent variable may have strongly correlated prediction errors.
Each
hidden variable can thus be interpreted as a separate unobserved teacher, or source of information,
and the classiﬁers that depend on it are diﬀerent perturbations of it.
Namely, even though we
observe m predictions for each instance, they are in fact generated by a hidden model with intrinsic
dimensionality K, where possibly K ≪m.
Let us now describe in detail our probabilistic model. First, since the latent variables α1, . . . , αK
follow the classical model of Dawid and Skene, their joint distribution is fully characterized by the
class imbalance b and the 2K probabilities
Pr(αk = 1|Y = 1)
and
Pr(αk = −1|Y = −1).
Next, we introduce an assignment function c : [m] →[K], such that if classiﬁer fi depends on αk
then c(i) = k. The dependence of classiﬁer fi on the class label Y is only through its latent variable
αc(i),
Pr(fi|αc(i), Y ) = Pr(fi|αc(i)).
(6)
Hence, classiﬁers fi, fj with c(i) ̸= c(j) maintain the original conditional independence assump-
tion of Eq. (4). In contrast, classiﬁers fi, fj with c(i) = c(j) are only conditionally independent
given αc(i),
Pr(fi = ai, fj = aj|αc(i)) = Pr(fi = ai|αc(i)) Pr(fj = aj|αc(i)).
(7)
Note that if the number of groups K is equal to the number of classiﬁers, then all classiﬁers are
conditionally independent, and we recover the original model of Dawid and Skene.
Since the model now consists of three layers, the remaining parameters to describe it are the
sensitivity ψα
i and speciﬁcity ηα
i of the i-th classiﬁer given its latent variable αc(i),
ψα
i =Pr(fi = 1|αc(i) = 1),
ηα
i =Pr(fi =−1|αc(i) =−1).
By Eq. (6), the overall sensitivity ψi of the i-th classiﬁer is related to ψα
i and ηα
i via
ψi = Pr(αc(i) = 1|Y = 1)ψα
i + Pr(αc(i) = −1|Y = 1)(1 −ηα
i ),
(8)
with a similar expression for its overall speciﬁcity ηi.
4

## Page 5

Remark on Model Identiﬁability.
Note that the model depicted in Fig. 1(right) is in general
not identiﬁable. For example, the classical model of Dawid and Skene can also be recovered with
a single latent variable K = 1, by having α1 = Y . Similarly, for a latent variable that has only a
single classiﬁer dependent on it, the parameters ψi, ηi and ψα, ηα are non-identiﬁable. Nonetheless,
these non-identiﬁability issues do not aﬀect our algorithms, described below.
Problem Formulation.
We consider the following totally unsupervised scenario. Let Z be a
binary m × n matrix with entries Zij = fi(xj), where fi(xj) is the label predicted by classiﬁer fi at
instance xj. We assume xj are drawn i.i.d. from pX(x). We also assume the m classiﬁers satisfy
our generalized model, but otherwise we have no prior knowledge as to the number of groups K, the
assignment function c or the classiﬁer accuracies (sensitivities ψi,ψα
i and speciﬁcities ηi, ηα
i ). Given
only the matrix Z of binary predictions and no labeled data, we consider the following problems:
1. Is it possible to detect strongly dependent classiﬁers, and estimate the number of groups and
the corresponding assignment function c?
2. Given a positive answer to the previous question, how can we estimate the sensitivities and
speciﬁcities of the m diﬀerent classiﬁers and construct an improved, possibly non-linear, meta
learner ?
3
Estimating the assignment function
The main challenge in our model is the ﬁrst problem of estimating the number of groups K and the
assignment function c. Once c is obtained, we will see in Section 4 that our second problem can
be reduced to the conditional independent case, already addressed in previous works [9,10,15,25].
In principle, one could try to ﬁt the whole model by maximum likelihood, however this results in a
hard combinatorial problem. We propose instead to ﬁrst estimate only K and c. We do so using
the low-rank structure of the covariance matrix of the classiﬁers, implied by our model.
The covariance matrix.
Let R denote the m×m population covariance matrix of the m classiﬁers
rij = E[(fi −E[fi])(fj −E[fj])].
(9)
The following lemma describes its structure. It generalizes a similar lemma, for the standard
Dawid and Skene model, proven in [15]. The proof of this and other lemmas below appear in the
appendix.
Lemma 1. There exists two vectors von, voff ∈Rm such that for all i ̸= j,
rij =

voff
i
· voff
j
if c(i) ̸= c(j)
von
i
· von
j
if c(i) = c(j)
(10)
The population covariance matrix is therefore a combination of two rank-one matrices. The
block diagonal elements i, j with c(i) = c(j) correspond to the rank-one matrix von(von)T , where on
stands for on-block, while the oﬀ-block diagonal elements, with c(i) ̸= c(j) correspond to another
rank-one matrix voff(voff)T . Let us deﬁne the indicator 1c(i, j)
1c(i, j) =
(
1
c(i) = c(j)
0
otherwise
(11)
The non-diagonal elements of R can thus be written as follows,
rij = 1c(i, j)von
i von
j
+ (1 −1c(i, j))voff
i
voff
j
.
(12)
5

## Page 6

Learning the model in the ideal setting.
It is instructive to ﬁrst examine the case where the
data is generated according to our model, and the population covariance matrix R is exactly known,
i.e. n = ∞. The question of interest is whether it is possible to recover the assignment function in
this setting.
To this end, let us look at the possible values of the determinant of 2x2 submatrices of R,
Mijkl = det
 rij
ril
rkj
rkl

(13)
Due to the low rank structure described in lemma 1, we have the following result, with the exact
conditions appearing in the appendix.
Lemma 2. Assume the two vectors von and voff are suﬃciently diﬀerent, then Mijkl = 0 if and only
if either: (i) Three or more of the indices i, j, k and l belong to the same group or (ii) c(i) ̸= c(j),
c(j) ̸= c(k), c(k) ̸= c(l) and c(l) ̸= c(i).
With details in the appendix, comparing the indices (j, k, l) where M(i1, j, k, l) = 0 with i1 ﬁxed,
to those where M(i2, j, k, l) = 0, we can deduce, in polynomial time, whether c(i1) = c(i2).
Learning the model in practice.
In practical scenarios, the population covariance matrix R
is unknown and we can only compute the sample covariance matrix ˆR. Furthermore, our model
would typically be only an approximation of the classiﬁers dependency structure. Given only ˆR,
the approach to recover the assignment function described above, based on exact matching of the
pattern of zeros of the determinants of various 2x2 submatrices is clearly not applicable.
In principle, since E[ ˆR] = R a standard approach would be to deﬁne the following residual
∆(von, voff, c) =
X
i̸=j
1c(i, j)(von
i von
j
−ˆrij)2 + (1 −1c(i, j))(voff
i
voff
j
−ˆrij)2,
(14)
and ﬁnd its global minimum. Unfortunately, as stated in the following lemma and proven in the
appendix, in general this is not a simple task.
Lemma 3. Minimizing the residual of Eq. (14) for a general covariance matrix ˆR is NP-hard.
In light of Lemma 3, we now present a tractable algorithm to estimate K and c and provide
some theoretical support for it. Our algorithm is inspired by the ideal setting which highlighted
the importance of the determinants of 2 × 2 submatrices. To detect pairs of classiﬁers fi, fj that
strongly violate the conditional independence assumption, we thus compute the following score
matrix ˆS = ˆS( ˆR),
ˆsij =
X
k,l̸=i,j
|ˆrijˆrkl −ˆrilˆrkj|.
(15)
The idea behind the score matrix is the following: Consider the score matrix S computed with the
population covariance R. Lemma 2 characterized the cases where the submatrices in Eq. (15) are
of rank-one, and hence their determinant is zero. When c(i) ̸= c(j) most submatrices come from
four diﬀerent groups, i.e. will have rank one, and thus the sum sij will be small. On the other hand,
when c(i) = c(j) many submatrices will not be rank one and thus sij will be large, assuming no
degeneracy between von and voff. As ˆS
n→∞
−−−−→S, large values of ˆsij serve as an indication of strong
conditional dependence between classiﬁers fi and fj.
The following lemma provides some theoretical justiﬁcation for the utility of the score matrix S
computed with the population covariance, in recovering the assignment function c. For simplicity, we
analyze the ’symmetric’ case where the class imbalance b = 0, Pr(αk = −1|y = −1) = Pr(αk = 1|y =
6

## Page 7

Algorithm 1 Estimating the assignment function c and vectors von, voff
1: Estimate the covariance matrix R (9).
2: Obtain the score matrix by (15)
3: for all 1 < k < m do
4:
Estimate c by performing spectral clustering with the Laplacian of the score matrix.
5:
Use the clustering function to estimate the two vectors von, voff.
6:
Calculate residual by (14).
7: end for
8: Pick the assignment function and vectors which yield minimal residual.
1) and all groups have equal size of m/K. We measure deviation from conditional independence by
the following matrices of conditional covariances C+ and C−,
c+
ij
=
E[(fi −E[fi])(fj −E[fj])|Y = 1]
c−
ij
=
E[(fi −E[fi])(fj −E[fj])|Y = −1].
(16)
Finally, we assume there is a δ > 0 such that the balanced accuracies of all classiﬁers satisfy
(2πi −1) > δ > 0.
Lemma 4. Under the assumptions described above, if c(i) = c(j) then
sij > m2

1 −3
K

δ2|c+
ij| = m2

1 −3
K

δ2|c−
ij|.
(17)
In contrast, if c(i) ̸= c(j) then
sij < 2m2
K

1 −2
K

.
(18)
An immediate corollary from lemma 4, is that if the classiﬁers are suﬃciently accurate, and their
dependencies within each group are strong enough then the score matrix exhibits a clear gap with
max
c(i)̸=c(j) Sij <
min
c(i)=c(j) Sij. In this case, even a simple single-linkage hierarchical clustering algorithm
can recover the correct assignment function from S. In practice, as only ˆS is available, we apply
spectral clustering which is more robust, and works better in practice.
We illustrate the usefulness of the score matrix using the DREAM challenge S1 dataset, which
contains m = 124 classiﬁers. Fig. 2a shows the matrix of conditional covariance 1
2(C+ + C−) of Eq.
(16), computed using the ground truth labels. Fig. 2b shows the score matrix ˆS computed using only
the classiﬁers predictions. We also plot the values of the score matrix vs. the conditional covariance
in ﬁgure 3. Clearly, a high score is a reliable indication for strong conditional dependencies between
classiﬁers.
It is important to note that the time complexity needed to build the score matrix S is O(m4).
While quartic scaling is usually considered too expensive, in our case as the number m of classiﬁers
in many real world problems is in the hundreds our algorithm can run on these datasets in less than
an hour. This can be sped-up, for example, by sampling the elements of S instead of computing the
full matrix [8].
Estimating the assignment function c.
We estimate c by spectral clustering the score matrix
ˆS of Eq. (15). As the number of clusters or groups K is unknown, we choose the one which minimizes
the residual function deﬁned in Eq. (14). The steps for estimating the number of groups K and the
assignment function c are summarized in Algorithm 1. Note that retrieving von and voff from the
7

## Page 8

20
40
60
80
100
120
20
40
60
80
100
120
−0.05
0
0.05
0.1
0.15
0.2
0.25
(a)
The
conditional
covariance
matrix
1
2(C+ + C−)
of
the
DREAM
dataset
S1,
computed using the ground truth labels.
 
 
20
40
60
80
100
120
20
40
60
80
100
120
40
60
80
100
120
140
160
180
200
220
240
(b) The score matrix
ˆS of the DREAM S1
dataset, computed from the matrix of classiﬁer
predictions. For visualization purposes, the up-
per limit of the above score matrix is ﬁxed at
300.
Fig. 2
0.05
0.1
0.15
0.2
0.25
50
100
150
200
250
300
1
2(C+ + C−)
S
Fig. 3: Values of S vs. the corresponding conditional covariance matrix 1
2(C++C−) for the DREAM
dataset S1. The blue dots represent the mean value, the upper and lower red dots represent the
20th and 80th quantiles, respectively.
covariance matrix is a rank-one matrix completion problem, for which several solutions exist, for
example see [4]. Also note that while we compute spectral clustering for various number of clusters,
the costly eigen-decoposition step only needs to be done once.
4
The latent spectral meta learner
Estimating the model parameters.
Given estimates of K and of the assignment function c,
estimating the remaining model parameters can be divided into two stages: (i) Estimating the sen-
sitivity and speciﬁcity of the diﬀerent classiﬁers given the latent variables αk: ψα
i , ηα
i (ii) Estimating
the probabilities associated with the latent variables, Pr(αk = 1|Y = 1) and Pr(αk = −1|Y = −1).
8

## Page 9

Algorithm 2 Estimate model parameters
1: Input: Matrix of predictions fi(xj), parameters K and c.
2: for k = 1, .., K do
3:
Find all classiﬁers fi where c(i) = k
4:
Estimate ψα
i , ηα
i and E[αk]
5:
Estimate the latent values αk(xj), ∀j = 1, ..., n
6: end for
7: Estimate Pr(αk = 1|Y = 1), Pr(αk = −1|Y = −1)
The key observation is that in each of these stages the underlying model follows the classical
conditional independent model of [5]. In particular, classiﬁers with a common latent variable are
conditionally independent given its value. Similarly the K latent variables themselves are condition-
ally independent given the true label Y . Thus, we can solve the two stages sequentially by any of
the various methods already developed for the Dawid and Skene model. In our implementation, we
used the spectral meta learner proposed in [9], whose code is publicly available. A pseudo-code for
this process appears in Algorithm 2.
Label Predictions.
Once all the parameters of the model are known, for each instance x we
estimate its label by maximum likelihood
ˆy = argmax
y=±1
Pr(f1(x), . . . , fm(x)|y).
(19)
Following our generative model, Fig.
1(right), the above probability is a function of the model
parameters b, ψα
i , ηα
i , ψα, ηα, and the assignment function c.
Classiﬁer selection.
In some cases, it is required to construct a sparse ensemble learner which
uses only a small subset of at most M out of the available m classiﬁers. This problem of selecting
a small subset of classiﬁers, known as ensemble pruning, has mostly been studied in supervised
settings, see [13,19,24].
Under the conditional independence assumption, the best subset simply consists of the M most
accurate classiﬁers. In our model, in contrast, the correlations between the classiﬁers have to be
taken into account. Assuming the required number of classiﬁers is smaller than the number of groups
M ≤K, a simple approach is to select the M most accurate classiﬁers under the constraint that
they all come from diﬀerent groups. This creates a balance between accuracy and diversity.
5
Experiments
We demonstrate the performance of the latent variable model on artiﬁcial data, on datasets from
the UCI repository and on the ICGA-TCGA dream challenge.
Throughout our experiments, we compare the performance of the following unsupervised ensem-
ble methods: (1) Majority voting, which serves as a baseline; (2) SML+EM - a spectral meta-learner
based on the independence assumption [9] which provides an initial guess, followed by EM iterations;
(3) Oracle-CI: A linear meta-learner based on Eq. (5), which assumes conditional independence
but is given the exact accuracies of all the individual classiﬁers. (4) L-SML (latent SML), the new
algorithm presented in this work.
For the artiﬁcial data, we also present the performance of its oracle meta-learner, denoted
Oracle-L, which is given the exact structure and parameters of the model, and predicts the la-
bel Y by maximum likelihood.
9

## Page 10

2
4
6
8
10
0.6
0.65
0.7
0.75
0.8
|G1|
πens
 
 
Vote
SML+EM
Oracle−CI
L−SML
Oracle−L
Fig. 4:
Simulated data:
Ensemble learner
balanced accuracy vs. the size of group 1.
Median
Max
Vote
SML
Oracle
L−SML
0.72
0.74
0.76
0.78
0.8
0.82
0.84
π
Fig. 5: UCI magic dataset, a comparison of
four unsupervised ensemble learners.
5.1
Artiﬁcial Data
To validate our theoretical analysis, we generated artiﬁcial binary data according to our assumed
model, on a balanced classiﬁcation problem with b = 0. We generated an ensemble of m = 20
classiﬁers with n = 104 instances. All the parameters of the ensemble were chosen uniformly at
random from the following intervals: Pr(α = 1|Y = 1), Pr(α = −1|Y = −1) ∈[0.5, 0.8], {ψα
i , ηα
i } ∈
[0.7, 0.9]. We consider the case where there is only one group G1 of correlated classiﬁers, with the
remaining m −|G1| classiﬁers all conditionally independent. The size of the correlated group |G1|
increases from 1 to 10. Note that for |G1| = 1 all classiﬁers are conditionally independent. Fig. 4
compares the balanced accuracy of the ﬁve unsupervised ensemble learners described above, as a
function of the size of the ﬁrst group, |G1|. As can be seen in Fig. 4, up to |G1| = 6, the ensemble
learner based on the concept of correlated classiﬁers achieves similar results to the optimal classiﬁer
(’oracle-L’). As expected from Lemma 4, as |G1| increases, it is harder to correctly estimate c with
the score matrix.
A complementary graph which presents the probability to recover the correct assignment function
as a function of |G1| appears in the appendix. As expected, the degradation in performance starts
when the algorithm fails to correctly estimate the model structure.
5.2
UCI data sets
We applied our algorithms on various binary classiﬁcation problems using 4 datasets from the UCI
repository: Magic, Spambase, Miniboo and Musk. Our ensemble of m = 16 classiﬁers consists of
4 random forests, 3 logistic model trees, 4 SVM and 5 naive Bayes. Each classiﬁer was trained on
a separate, randomly chosen labeled dataset. In our unsupervised ensemble scenario we had access
only to their predictions on a large independent test set.
We present results for the magic dataset, which contains 19000 instances with 11 attributes. The
task is to classify each instance as background or high energy gamma rays. Further details and
results on the other datasets appear in the appendix.
As seen in Fig. 5, the L-SML improves substantially over the standard SML, and even on the
oracle classiﬁer that assumes conditional independence.
Our method also outperforms the best
individual classiﬁer.
In the appendix we show the conditional covariance matrix, Fig. 8, and our assignment, Fig. 9.
It can be observed that strongly dependent classiﬁers are indeed grouped together correctly.
10

## Page 11

Mean
Best
Vote
SML+
EM
Oracle-
CI
L-SML
S1
6.1
1.7
2.8
1.7
1.7
1.6
S2
8.7
1.8
4.0
2.8
2.8
2.3
S3
8.3
2.5
4.3
2.3
2.3
1.8
Table 1: Balanced error of meta-classiﬁers based on the full ensemble. For reference, the ﬁrst two
columns give the mean and smallest balanced error of all classiﬁers.
Vote
SML+EM
Oracle-CI
L-SML
S1
3.2
2.3
1.9
2.0
S2
4.3
4.1
2.5
2.8
S3
2.9
2.9
2.8
2.5
Table 2: Balanced error of sparse meta-classiﬁers.
5.3
The DREAM mutation calling challenge
The ICGC-TCGA DREAM challenge is an international eﬀort to improve standard methods for
identifying cancer-associated mutations and rearrangements in whole-genome sequencing (WGS)
data. This publicly available database contains both real and synthetic in-silico tumor instances.
The database contains 14 diﬀerent datasets, each with over 100,000 instances.
Participants in the currently open competition are given access to the predictions of about a
hundred diﬀerent classiﬁers (denoted there as pipe-lines)1. These classiﬁers were constructed by
various labs worldwide, each employing their own biological knowledge and possibly proprietary
labeled data. The two current challenges are to construct a meta-learner, by using either (1) all m
classiﬁers; or (2) at most ﬁve of them. We evaluate the performance of the diﬀerent meta-classiﬁers
fmc by their balanced error,
1 −π = 1
2(Pr(fmc = 1|y = −1) + Pr(fmc = −1|y = 1)).
Below we present results on the datasets S1, S2 and S3 for which ground-truth labels have been
released.
Challenge I.
The balanced errors of the diﬀerent meta-learners, constructed using all m classiﬁers,
are given in table 1.
The L-SML method outperforms the other meta-learners in all the three
datasets. On the S3 dataset, it reduces the balanced error by more than 20 % over competing meta
learners.
Challenge II
Here the goal is to construct a sparse meta-learner based on at most ﬁve individual
classiﬁers from the ensemble. For the methods based on the Dawid and Skene model (SML+EM,
voting and Oracle-CI), we took the 5 classiﬁers with the highest estimated (or known) balanced
accuracies. For our model, since the estimated number of groups is larger than ﬁve, we ﬁrst took the
best classiﬁer from each group, and then chose the ﬁve classiﬁers with highest estimated balanced
accuracies.
For all methods, the ﬁnal prediction was made by a simple vote of the ﬁve chosen
classiﬁers. Though potentially sub-optimal, we nonetheless chose it as our purpose was to compare
the diversity of the diﬀerent classiﬁers.
The results presented in table 2 show that our method outperforms voting and SML, and are
similar to those achieved by the oracle learner.
1The data can be downloaded from the challenge website http://dreamchallenges.org/
11

## Page 12

6
Acknowledgments
This research was funded in part by the Intel Collaborative Research Institute for Computational
Intelligence (ICRI-CI). Y.K. is supported by the National Institutes of Health Grant R0-1 CA158167,
and R0-1 GM086852.
12

## Page 13

References
[1] N. Aghaeepour, G. Finak, H. Hoos, T.R. Mosmann, R. Brinkman, R. Gottardo, R.H. Scheuer-
mann, FlowCAP Consortium, and DREAM Consortium. Critical assessment of automated ﬂow
cytometry data analysis techniques. Nature methods, 10(3):228–238, 2013.
[2] A. Anandkumar, R. Ge, D. Hsu, S.M. Kakade, and M. Telgarsky. Tensor decompositions for
learning latent variable models. Journal of Machine Learning Research, 15:2773–2832, 2014.
[3] P.C. Boutros, A.A. Margolin, J.M. Stuart, A. Califano, and G. Stolovitzky.
Toward bet-
ter benchmarking: challenge-based methods assessment in cancer genomics. Genome biology,
15(9):462, 2014.
[4] E.J. Cand`es and B. Recht. Exact matrix completion via convex optimization. Foundations of
Computational Mathematics, 9:717–772, 2009.
[5] A. P. Dawid and A. M. Skene. Maximum likelihood estimation of observer error-rates using the
EM algorith. Journal of the Royal Statistical Society. Series C, 28:20–28, 1979.
[6] P. Donmez, G. Lebanon, and K. Balasubramanian. Unsupervised supervised learning i: Es-
timating classiﬁcation and regression errors without labels. Journal of Machine Learning Re-
search, 11:1323–1351, 2010.
[7] A.D. Ewing, K.E. Houlahan, Y. Hu, K. Ellrott, C. Caloian, T.N. Yamaguchi, J.Ch. Bare,
C. P’ng, D. Waggott, and V.Y. Sabelnykova. Combining tumor genome simulation with crowd-
sourcing to benchmark somatic single-nucleotide-variant detection. Nature methods, 12:623,
2015.
[8] E. Fetaya, O. Shamir, and S. Ullman. Graph approximation and clustering on a budget. 18th
conference on artiﬁcial intelligence and statistics, 2015.
[9] A. Jaﬀe, B. Nadler, and Y. Kluger. Estimating the accuracies of multiple classiﬁers without
labeled data. In 18th conference on artiﬁcial intelligence and statistics, pages 407–415, 2015.
[10] P. Jain and S. Oh. Learning mixtures of discrete product distributions using spectral decom-
positions. Journal of Machine Learning Research, 35:1–33, 2014.
[11] D.R. Karger, S. Oh, and D. Shah. Budget-optimal crowdsourcing using low-rank matrix ap-
proximations. In IEEE Alerton Conference on Communication, Control and Computing, pages
284–291, 2011.
[12] J. A. Lee. Click to cure. The Lancet Oncology, 2013.
[13] G. Martinez-Muoz, D. Hern´andez-Lobato, and A. Suarez. An analysis of ensemble pruning
techniques based on ordered aggregation. Pattern Analysis and Machine Intelligence, IEEE
Transactions on, 31(2):245–259, 2009.
[14] M. Micsinai, F. Parisi, F. Strino, P. Asp, B.D. Dynlacht, and Y. Kluger. Picking chip-seq peak
detectors for analyzing chromatin modiﬁcation experiments. Nucleic acids research, 2012.
[15] F. Parisi, F. Strino, B. Nadler, and Y. Kluger. Ranking and combining multiple predictors
without labeled data. Proceedings of the National Academy of Sciences, 111:1253–1258, 2014.
[16] E.A. Platanios, A. Blum, and T. Mitchell.
Estimating accuracy from unlabeled data.
In
Uncertainty in Artiﬁcial Intelligence, 2014.
13

## Page 14

[17] A.J. Quinn. Crowdsourcing decision support: frugal human computation for eﬃcient decision
input acquisition. PhD thesis, 2014.
[18] V.C. Raykar, Y. Shipeng, L.H. Zhao, G.H. Valdez, C. Florin, L. Bogoni, and Moy L. Learning
from crowds. J. Machine Learning Research, 11:1297–1322, 2010.
[19] L. Rokach. Collective-agreement-based pruning of ensembles. Computational Statistics & Data
Analysis, 53(4):1015–1026, 2009.
[20] A. Sheshadri and M. Lease. Square: A benchmark for research on computing crowd concensus.
In AAAI conference on human computation and crowdsourcing, 2013.
[21] Tian Tian and Jun Zhu. Uncovering the latent structures of crowd labeling. In Advances in
Knowledge Discovery and Data Mining, pages 392–404. Springer, 2015.
[22] P. Welinder, S. Branson, S. Belongie, and P. Perona. The multidimensional wisdom of crowds.
In Advances in Neural Information Processing Systems 23 (NIPS 2010), 2010.
[23] J. Whitehill, P. Ruvolo, T. Wu, J Bergsma, and J.R. Movellan. Whose vote should count
more: Optimal integration of labels from labelers of unknown expertise. In Advances in Neural
Information Processing Systems 22 (NIPS 2009), 2009.
[24] Xu-Cheng Yin, Kaizhu Huang, Chun Yang, and Hong-Wei Hao. Convex ensemble learning with
sparsity and diversity. Information Fusion, 2014.
[25] Y. Zhang, X. Chen, D. Zhou, and M.I. Jordan. Spectral methods meet EM: A provably optimal
algorithm for crowdsourcing. In Advances in Neural Information Processing Systems, volume 27,
pages 1260–1268, 2014.
14

## Page 15

A
Proof of Lemma 1
This proof is based on the following lemma, which appears in [15]:
If two classiﬁers fi, fj are conditionally independent given the class label Y , then the covariance
between them is equal to,
rij = (1 −b2)(ψi + ηi −1)(ψj + ηj −1).
(20)
In our model, if c(i) ̸= c(j), then fi, fj are indeed conditionally independent (Fig. 1,right). The
ﬁrst part of lemma 1 follows directly from Eq. (20), with voff
i
=
√
1 −b2(ψi + ηi −1).
To prove the second part of lemma 1, we note that according to our model, two classiﬁers fi, fj
with c(i) = c(j) are conditionally independent given the value of their latent variable α. Therefore,
we can treat α as the class label, and apply Eq. (20) with b replaced by the expectation of α, and
the sensitivity and speciﬁcity ψi, ηi replaced by ψα
i , ηα
i respectively. Hence, Eq. (20) becomes,
rij = (1 −E[α]2)(ψα
i + ηα
i −1)(ψα
j + ηα
j −1) = von
i von
j ,
(21)
where von
i
=
p
1 −E[α]2(ψα
i + ηα
i −1).
B
Proof of Lemma 2
We assume that von and voff are suﬃciently diﬀerent in the following precise sense: We require that
for all 4 distinct indices i, j, k, l, von
i
· von
j
· von
k · von
l
̸= voff
i
· voff
j
· voff
k
· voff
l
.
Next, we elaborate on the relation between voff
i
and von
i . Let us denote by ψy
α, ηy
α the sensitivity
and speciﬁcity of the latent variable α. Let fi be a classiﬁer that depends on α. Applying Bayes
rule, its overall sensitivity and speciﬁcity is given by,
ψi = ψy
αψα
i + (1 −ψy
α)(1 −ηα
i )
ηi = ηy
αηα
i + (1 −ηy
α)(1 −ψα
i ).
(22)
Adding ψi and ηi we get the following,
ψi + ηi −1 = (ψy
α + ηy
α −1)(ψα
i + ηα
i −1).
(23)
If c(i) = c(j) we have the following dependency between (voff
i
, voff
j
) and (von
i , von
j ),
"
voff
i
voff
j
#
=
p
1 −b2(ψy
α + ηy
α −1)
 (ψα
i + ηα
i −1)
(ψα
j + ηα
j −1)

=
s
1 −b2
1 −E[α]2 (ψy
α + ηy
α −1)
 von
i
von
j

. (24)
It follows that two elements voff
i
, voff
j
where c(i) = c(j) are linearly dependent with the correspond-
ing elements of von
i , von
j . This fact shall be useful in proving the lemma.
To prove lemma 2 we analyze all various possibilities for the group assignments of the four indices
i, j, k, l of
M(i, j, k, l) = det
 rij
ril
rkj
rkl

.
1. c(i) = c(j) = c(k) = c(l): In this case M(i, j, k, l) = von
i von
j von
k von
l
−von
i von
l von
k von
j
= 0.
2. c(i) ̸= c(j), and c(j) ̸= c(k), and c(k) ̸= c(l) and c(l) ̸= c(i): Here M(i, j, k, l) = voff
i
voff
j
voff
k
voff
l
−
voff
i
voff
l
voff
k
voff
j
= 0.
15

## Page 16

3. c(i) = c(l) = c(k) ̸= c(j): M(i, j, k, l) = voff
i
voff
j
von
k von
l −von
i von
l voff
k
voff
j
= voff
j
von
l

voff
i
von
k −von
i voff
k

.
From the linear dependency shown in Eq. (24).

voff
i
von
k −von
i voff
k

= 0.
4. c(i) = c(j), c(k) = c(l) and c(i) ̸= c(k): M(i, j, k, l) = von
i von
j von
k von
l
−voff
i
voff
l
voff
k
voff
j
̸= 0
from our assumption.
It can be seen that Mijkl is equal to zero only if either three or more of the indices are equal (cases
(1) and (2)) or all four pairs which appear in the determinant belong to diﬀerent groups (case (3)).
C
Algorithm for the ideal setting
An immediate conclusion from lemma 2, is that the indices i, j, k and l for which M(i, j, k, l) = 0
depend only on the assignment function.
This means we can compare the pattern of zeros for
M(i1, j, k, l) and M(i2, j, k, l) to decide if fi1 and fi2 belong to the same group. If c(i1) = c(i2)
then M(i1, j, k, l) = 0 ⇐⇒M(i2, j, k, l) = 0. On the other hand if c(i1) ̸= c(i2) and at least one
of the indices i1 and i2 , w.l.o.g i1, belongs to a group with more than one element, then we can
ﬁnd j, k and l such that M(i1, j, k, l) ̸= 0 but M(i2, j, k, l) = 0. This occurs when c(i1) = c(j), and
c(i2) ̸= c(j) ̸= c(k) ̸= c(l).
This means that by comparing the pattern of zeros, we can recover the assignment function.
Notice, that according to the algorithm, all singleton classiﬁers, that is, classiﬁers who are condi-
tionally independent with the rest of the ensemble, are grouped together under a common latent
variable. This is not a problem, as our model is not unique and this is an equivalent probabilistic
model, when the latent variable being identical to Y .
Algorithm 3 Check if c(i1) = c(i2)
1: Initialize (m −2) × (m −3) × (m −4) arrays T1, T2 to zero
2: for j ̸= k ̸= l ̸= i1, i2 do
3:
if ri1jrkl −ri1lrkj = 0 then (T1(j, k, l) = 1)
4:
end if
5:
if ri2jrkl −ri2lrkj = 0 then (T2(j, k, l) = 1)
6:
end if
7: end for
8: if (T1 = T2) then
9:
c(i1) = c(i2).
10: else
11:
c(i1) ̸= c(i2).
12: end if
D
Minimizing ∆is a NP hard problem
We prove lemma 3 for the case of K = 2 clusters and known voff,von vectors. Our goal is to ﬁnd a
minimizer for the following residual:
ˆc = argmin
c
∆(c) = argmin
c
X
i,j
1c(i, j)(von
i von
j
−rij)2 + (1 −1c(i, j))(voff
i
voff
j
−rij)2
(25)
For the case of K = 2 we can simplify the residual considerably. Let us deﬁne a vector x ∈{−1, 1}m
where xi = 1 if c(i) = 1 and xi = −1 if c(i) = 2. We can replace the indicator function 1(i, j) with
16

## Page 17

the following,
1(i, j) = (1 + xixj)
2
,
1 −1(i, j) = (1 −xixj)
2
.
(26)
In addition, we can replace the minimization over c with a minimization over x,
ˆx
=
argmin
x
X
i,j
(1 + xixj)
2
(von
i von
j
−rij)2 + (1 −xixj)
2
(voff
i
voff
j
−rij)2
=
argmin
x
X
i,j
1
2

(von
i von
j
−rij)2 + (voff
i
voff
j
−rij)2
+xixj
2

(von
i von
j
−rij)2 + (voff
i
voff
j
−rij)2
.
(27)
The ﬁrst term does not depend on x and we can omit it from the minimization problem. Let us also
deﬁne the matrix ˜R,
˜rij =

(von
i von
j
−rij)2 + (voff
i
voff
j
−rij)2
2
(28)
We are left with the following minimization problem:
ˆx = argmin
x
X
i,j
xixj ˜rij = argmin
x
x′ ˜Rx
(29)
If there is a binary vector whose residual is precisely zero, then it can be found by computing the
eigenvector with smallest eigenvalue of the matrix ˜R. If, however, the minimal residual is not zero,
then eq. (29) is a quadratic optimization problem involving discrete variables, which is well known
to be a NP-hard problem.
E
Proof of Lemma 4
We start by proving the ﬁrst part of the lemma, where c(i) = c(j). The score matrix sij is a sum
of all possible 2 × 2 determinants,
si,j =
X
k,l̸=i,j
|rijrkl −rilrjk| =
X
k,l̸=i,j
skl
ij ,
(30)
where we deﬁne skl
ij as a single score element. The following table separates the group of skl
ij score
elements into three types, and states the number of elements in each type.
Element type
Number of elements
c(i) = c(j) ̸= c(k) ̸= c(l)
m2  1 −3
K +
2
K2

c(i) = c(j) = c(k) ̸= c(l)
m2   1
K −2
m
  1 −1
m

c(i) = c(j) = c(k) = c(l)
m2   1
K −2
m
   1
K −3
m

According to lemma 1, the contribution to the score from elements of the second and third type
is exactly 0 (see details in Sec. B). We will therefore focus on analyzing the score elements of the ﬁrst
type, where c(i) = c(j) ̸= c(k) ̸= c(l). Recall, that we assume a symmetrical case where b = 0, and
Pr(α = 1|y = 1) = Pr(α = −1|y = −1). These assumptions imply that E[αk] = 0 for all k = 1...K.
Let us consider Lem. 1 in order to analyze the value of skl
ij,
skl
ij
=
|rijrkl −rijrjk| = |(2πα
i −1)(2πα
j −1)(2πk −1)(2πl −1) −(2πi −1)(2πj −1)(2πk −1)(2πl −1)|
=
|(2πk −1)(2πl −1)
 (2πα
i −1)(2πα
j −1) −(2πi −1)(2πj −1)

|
(31)
17

## Page 18

where πα
i = 1
2(ψα
i + ηα
i ). For simplicity of notation, let us denote by γ the ratio of true positives
and negatives of the latent variables:
γ = Pr(αk = 1|Y = 1) = Pr(αk = −1|Y = −1)
(32)
It can easily be shown that the following holds:
(2πi −1) = (2γ −1)(2πα
i −1)
(2πj −1) = (2γ −1)(2πα
j −1)
(33)
Inserting (33) into (31) we get,
skl
ij = |(2πk −1)(2πl −1)(2πα
i −1)(2πα
j −1)(1 −(2γ −1)2)| =
|4(2πk −1)(2πl −1)(2πα
i −1)(2πα
j −1)(γ(1 −γ))|
(34)
Let us now derive the values of the conditional covariance matrices C+, C−. In order to obtain
C+, we can apply the ﬁrst part of Lem.1, and replace the class imbalance b, which is the mean value
of Y , with E[α|Y = 1]. A similar argument applies to C−. The value for the conditional expectation
of α is equal to,
E[α|Y = 1] = 2γ −1
E[α|Y = −1] = 1 −2γ
(35)
A simple derivation yields the following for both cases,
(1 −E[α|Y = 1]2) = (1 −E[α|Y = −1]2) = 4γ(1 −γ)
(36)
The value of c+
ij is therefor equal to c−
ij, and both are equal to the following,
c+
ij = c−
ij = 4γ(1 −γ)(2πα
i −1)(2πα
j −1)
(37)
Inserting (37) into (34) we get the following,
skl
ij = |(2πk −1)(2πl −1)c+
ij| = |(2πk −1)(2πl −1)c−
ij|
(38)
We will remain with C+ for simplicity, The total score contribution of the ﬁrst type of elements is
therefore,
X
k,l
skl
ij = |c+
ij|
X
k,l
|(2πk −1)(2πl −1)|
(39)
Assuming (2πi −1) > δ > 0, ∀i, the latter simpliﬁes to,
sij > |c+
ij|δ2m2(1 −3
K +
2
K2 ) > |c+
ij|δ2m2(1 −3
K )
(40)
We next turn to proving an upper bound when c(i) ̸= c(j). Once again we can separate the
diﬀerent elements into three types,
Element type
Number of elements
c(i) ̸= c(j) ̸= c(k) ̸= c(l)
m2  1 −5
K +
6
K2

c(i) ̸= c(j) = c(k) ̸= c(l)
2m2   1
K −1
m
  1 −2
K

c(i) ̸= c(j) = c(k) = c(l)
m2   1
K −2
m
   1
K −3
m

The only contribution comes from the second type, as according to our model, if all indices come
from diﬀerent groups, or if three come from the same group, the determinant is equal to 0 (see.
B). In addition, since (2πi −1) > δ > 0 ∀i, the values of rij are positive for all (i, j) pairs. Since
0 < rij ≤1 for all score elements skl
ij = |rijrkl −rilrkj| ≤1. The total value of sij is bounded by the
following
sij ≤2m2
 1
K −1
m
 
1 −2
K

< 2m2
K

1 −2
K

(41)
18

## Page 19

2
4
6
8
10
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
|G1|
Pr(ˆc = c)
Fig. 6: Probability of estimating the exact
assignment function c as a function of the
size of the correlated group |G1|
2
4
6
8
10
−0.005
0
0.005
0.01
0.015
0.02
0.025
0.03
0.035
0.04
|G1|
MSE({ψ,η}m
i=1)
 
 
Vote
SML+EM
L−SML
Fig. 7: A comparison of the mean squared
error in estimating the accuracies of the dif-
ferent classiﬁers in the ensemble.
F
Additional results
F.1
Artiﬁcial data
In Fig. 6 we present the probability of our spectral clustering based algorithm to recover both the
correct number of classes K and the correct assignment function c, as a function of |G1|. Up to
|G1| = 6, our algorithm successfully estimates c, with no errors. When |G1| > 7, the algorithm
completely fails. The degradation in performance presented in Fig. 4, corresponds to the point
where the algorithm fails to estimate c correctly.
In Fig. 7 we present the mean squared error (MSE) of the sensitivity and speciﬁcity estimation
for the ensemble of classiﬁers, as a function of |G1|, deﬁned as
MSE({ψ, η}m
i=1) =
1
2m
m
X
i=1
 ( ˆψi −ψi)2 + (ˆηi −ηi)2
.
(42)
We compare the following three methods: (1) Majority vote ; (2) SML+EM; (3)L-SML. It can
be seen that the performance of the SML degrades very fast when the conditional independence
assumption is violated. The performance of the L-SML is almost perfect up to the point where
|G1| = 6, where as we have seen in Fig. 6, the model is correctly estimated. The performance is still
superior to other methods, even for large values of |G1|.
F.2
UCI results
For the magic dataset, Fig. 8 presents the conditional covariance matrix 1
2(C+ + C−), which is
unknown to us. The group of SVM classiﬁers (12-16) are highly dependent, as well as the group of
naive Bayes classiﬁers (8-11). The groups of random forest classiﬁers and logistic model trees are
weakly dependent.
Fig. 9 presents an example of the estimated assignment function ˆc for the same dataset. The
groups of SVM classiﬁers were assigned together, as well as the naive Bayes classiﬁers. Except for a
single pair, the random forest and logistic model trees were assigned to separate groups.
In ﬁgures 10a,10b and 10c we present the results for the following 3 additional datasets from the
UCI repository:
• Musk dataset - detection of certain types of molecules.
• Spam dataset - detection of spam from regular mail.
19

## Page 20

2
4
6
8
10
12
14
16
2
4
6
8
10
12
14
16
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Fig. 8: Magic database - conditional covari-
ance matrix 1
2(C+ + C−).
 
 
2
4
6
8
10
12
14
16
2
4
6
8
10
12
14
16
0
0.2
0.4
0.6
0.8
1
Fig. 9: Magic database - The estimated group
return by our algorithm.
• Miniboo dataset - detection of electron neutrinos (signal) from muon neutrinos (background).
The base classiﬁers are identical to the ones used for the Magic dataset: (1) 4 random forest (2)
3 Logistic Model Trees (3) 4 SVM (4) 5 naive Bayes .
In ﬁgures 10a-10d, the x-axis is the L-SML balanced error, and the y-axis is the SML balanced
error. The results of multiple experiments, each time with the classiﬁers constructed using diﬀerent
random subset of labeled examples, are presented as blue dots, while the red line represents the
y = x line, i.e. when the error of the L-SML and SML are the same. For the Magic dataset, ﬁgure
10d, we add two lines which represent 2% and 4% improvement over the standard SML.
We can see in the ﬁgures that the improvement due to explicit modeling of possible classiﬁer
dependencies is consistent across all datasets. The amount of improvement changes, however from
dataset to dataset. The following table presents a summary of the diﬀerent properties of the datasets
together with the average improvement in the balanced accuracy between the two methods.
Dataset
Number of instances
number of features
Mean diﬀerence
Magic
19000
11
4%
Spam
4600
57
0.5%
Miniboo
130000
50
0.2%
Musk
6600
168
4.7%
20

## Page 21

0.1
0.105
0.11
0.115
0.12
0.125
0.04
0.05
0.06
0.07
0.08
0.09
0.1
0.11
0.12
0.13
1 −πSML
1 −πL−SML
(a) UCI Musk dataset, a comparison between the
balanced error of the SML and L-SML.
0.056
0.058
0.06
0.062
0.045
0.05
0.055
0.06
0.065
1 −πSML
1 −πL−SML
(b) UCI Spambase dataset, a comparison be-
tween the balanced error of the SML and L-SML.
0.1
0.105
0.11
0.115
0.12
0.09
0.095
0.1
0.105
0.11
0.115
0.12
0.125
1 −πSML
1 −πL−SML
(c) UCI Miniboo dataset, a comparison between
the balanced error of the SML and L-SML.
0.19
0.195
0.2
0.205
0.21
0.215
0.15
0.16
0.17
0.18
0.19
0.2
0.21
0.22
1 −πens
1 −πens
(d) UCI Magic dataset. The magenta and green
lines represent 2% and 4% balanced accuracy im-
provement over the SML results.
Fig. 10
21
