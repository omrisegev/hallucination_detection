---
source_pdf: papers/A Deep Learning Approach to Unsupervised Ensemble Learning.pdf
slug: a-deep-learning-approach-to-unsupervised-ensemble-learning
pages: 24
extracted_on: 2026-07-13
---

# A Deep Learning Approach to Unsupervised Ensemble Learning

## Page 1

A Deep Learning Approach to Unsupervised Ensemble Learning
Uri Shaham1, Xiuyuan Cheng2, Omer Dror3, Ariel Jaﬀe3, Boaz Nadler3, Joseph
Chang1, and Yuval Kluger4
1Department of Statistics, Yale University
2Program of Applied Mathematics, Yale university
3Faculty of Mathematics and Computer Science, Weizmann Institute
4Department of Pathology, Yale University
Abstract
We show how deep learning methods can be applied in the context of crowdsourcing
and unsupervised ensemble learning.
First, we prove that the popular model of Dawid
and Skene, which assumes that all classiﬁers are conditionally independent, is equivalent
to a Restricted Boltzmann Machine (RBM) with a single hidden node. Hence, under this
model, the posterior probabilities of the true labels can be instead estimated via a trained
RBM. Next, to address the more general case, where classiﬁers may strongly violate the
conditional independence assumption, we propose to apply RBM-based Deep Neural Net
(DNN). Experimental results on various simulated and real-world datasets demonstrate
that our proposed DNN approach outperforms other state-of-the-art methods, in particular
when the data violates the conditional independence assumption.
1
Introduction
In recent years, crowdsourcing applications gained signiﬁcant popularity, and consequently
much academic attention. At the same time, deep learning has become a major tool in ma-
chine learning and artiﬁcial intelligence, demonstrating impressive performance in several ap-
plications, including computer vision, speech recognition and natural language processing.
The goal of this paper is to show that deep learning methods can also be applied to the
areas of crowdsourcing and unsupervised ensemble learning, and provide state-of-the-art results.
In unsupervised ensemble learning, one is given the predictions of d classiﬁers on a set of n
instances and the goal is to recover the true, unknown label of each instance. Dawid and Skene
(1979) were among the ﬁrst to consider such a setup. They assumed that the classiﬁers are
conditionally independent given the true labels. We refer to this model as the DS model and
also as the Conditional Independence model.
Despite its simplicity, computing the maximum likelihood estimates of the classiﬁers’ accu-
racies and the true labels in the DS model is a non-convex optimization problem. In their paper,
Dawid and Skene estimated these quantities by the EM algorithm, which is only guaranteed
1
arXiv:1602.02285v1  [stat.ML]  6 Feb 2016

## Page 2

to converge to a local optimum. In recent years, several authors developed computationally
eﬃcient spectral methods that are asymptotically consistent under the DS model, see Zhang
et al. (2014); Parisi et al. (2014); Jain and Oh (2013); Jaﬀe et al. (2014) and references therein.
The model of Dawid and Skene relied on two key assumptions that typically do not hold
in practice: (i) that classiﬁers make perfectly independent errors; and (ii) that these errors are
uniformly distributed across all instances. To address the second issue above, several authors
proposed richer models, that include parameters such as instance diﬃculty and varying skills
of annotators across diﬀerent regions of the input space, see for example Raykar et al. (2010),
Whitehill et al. (2009) and Welinder et al. (2010).
In contrast, relatively few works considered relaxations of the conditional independence
assumption: Platanios et al. (2014) proposed to estimate the accuracies of possibly dependent
classiﬁers, via their agreement rates over classiﬁer groups of diﬀerent sizes. Donmez et al. (2010)
proposed a model with pairwise interactions between all classiﬁers. Closest to our approach is
the work of Jaﬀe et al. (2015), who assumed that some of the classiﬁers may be conditionally
dependent, yet their dependency structure can be accurately described by a tree of depth 2.
In this manuscript, we propose a deep learning approach to unsupervised ensemble learning
problems with possibly dependent classiﬁers, where the conditional independence assumption
is strongly violated. We make the following contributions. First, we show that the DS model
has an equivalent parametrization in terms of a Restricted Boltzmann Machine (RBM) with a
single hidden node. Hence, under this model, the posterior probability of the true labels can be
estimated from a trained RBM. Next, to tackle violations of conditional independence, we show
how a RBM-based Deep Neural Net (DNN) can be applied to unsupervised ensemble learning,
and propose a heuristic for determining the DNN architecture. Experimentally, we compare our
approach to several state-of-the-art methods that are based on the conditional independence
assumption and relaxations of it. We show that our DNN approach often performs better than
the other methods on both simulated and real world datasets. Remarkably, we demonstrate
that in some cases, while the raw representation of the data contains correlated features, the
learned features in the last hidden layer are almost perfectly uncorrelated.
The structure of this manuscript is as follows: in Section 2 we give a formal deﬁnition of the
problem. A brief background on RBMs is given in Section 3. In Section 4 we show how RBMs
can be used to predict the true labels, under the assumption of conditional independence.
In Section 5 we describe how to estimate the labels using a RBM-based DNN. Experimental
results are reported in Section 6. The manuscript concludes with a brief summary in Section 7.
Proofs appear in the appendix.
1.1
Notation
Throughout this manuscript, X, H, Y are random variables, pθ, pλ are probability densities,
parametrized by θ, λ, respectively. We think of pθ as the distribution generating the data and
of pλ as the RBM model distribution. When the context is clear, we occasionally write p(x) as
a shorthand for p(X = x). The dimensions of the input data and the sample size are denoted
2

## Page 3

Y
X1
Xi
Xd
ψ1, η1
ψi, ηi
ψd, ηd
Figure 1: The conditional independence model, studied by Dawid and Skene (1979).
by d and n, respectively. We use σ(·) to denote the sigmoid function
σ(z) =
1
1 + e−z .
(1)
2
Problem Setup
Let X ∈{0, 1}d, Y ∈{0, 1} be random variables. We refer to Y as the label of X. The pair
(X, Y ) has a joint distribution, parametrized by θ and denoted by pθ(X, Y ), which is given by
pθ(X, Y ) = pθ(Y )pθ(X|Y ).
The joint distribution pθ(X, Y ) is not known to us, and neither are the marginals pθ(X), pθ(Y ).
Let (x(1), y(1)), . . . , (x(n), y(n)) be n i.i.d samples from pθ(X, Y ).
In unsupervised ensemble
learning, we observe x(1), . . . , x(n) and the learning task is to recover y(1), . . . , y(n). In this
application, the binary vector X = (X1, . . . , Xd)T contains the predictions of d classiﬁers or
annotators on an instance, whose label Y is unobserved.
2.1
The Conditional Independence Model
In their seminal paper, Dawid and Skene (1979), assumed that the conditional distribution
pθ(X|Y ) factorizes, i.e.,
pθ(X|Y ) ≡
d
Y
i=1
pθ(Xi|Y ).
(2)
Eq. (2), also known as the conditional independence model, is depicted in Figure 1. It is fully
parametrized by θ = ({ψi : i = 1, ..., d}, {ηi : i = 1, ..., d}, π), where
ψi = Pr(Xi = 1|Y = 1), ηi = Pr(Xi = 0|Y = 0),
π = Pr(Y = 1).
ψi, ηi are often referred to as sensitivity and speciﬁcity, respectively. Under the interpretation of
the Xi’s being classiﬁers, the sensitivity and speciﬁcity quantify the competence of the classiﬁers
or annotators and the conditional independence assumption means that all d classiﬁers make
independent errors.
3

## Page 4

The conditional independence model is often overly simplistic. In this manuscript we pro-
pose to apply deep learning techniques, speciﬁcally RBM-based DNNs, for unsupervised ensem-
ble learning problems, where the conditional independence is not likely to hold. The following
section gives essential background on RBMs, section 4 shows that a RBM with a single hidden
node is equivalent to the conditional independence model, and section 5 presents our RBM-
based DNN approach.
3
Restricted Boltzmann Machines
A Restricted Boltzmann Machine (RBM) is an undirected bipartite graphical model, consisting
of a set X of d visible binary random variables and a set H of m hidden binary random variables,
arranged in two layers, which are fully connected to each other. An illustration of a RBM is
depicted in Figure 2. A RBM is parametrized by λ = (W, a, b), where W is the weight matrix of
H1
Hm
X1
Xi
Xd
w11
w1i
w1d
Figure 2: A RBM with d visible and m hidden units.
the connections between the visible and hidden units, and a, b are the bias vectors of the visible
and hidden layers, respectively. Each conﬁguration (X = x, H = h) of a RBM is associated
with the following energy
Eλ(x, h) = −(aT x + bT h + xT Wh)
(3)
which deﬁnes the probability of the conﬁguration
pλ(X = x, H = h) = e−Eλ(x,h)
Z
,
where Z ≡P
x,h e−Eλ(x,h) is the partition function. The bipartite structure of the RBM implies
factorial conditional probabilities
pλ(X|H) =
Y
i
pλ(Xi|H),
pλ(H|X) =
Y
j
pλ(Hj|X),
given by
pλ(Xi = 1|H) = σ(ai + Wi.H)
pλ(Hj = 1|X) = σ(bj + XT W.j),
4

## Page 5

where σ(z) is the sigmoid function deﬁned in equation (1), Wi. is the i-th row of W and W.j is
its j-th column.
Given iid training data x(1), .., x(n)
∼pθ(X), the RBM parameters λ = (W, a, b) are
typically tuned to maximize the log-likelihood of the training data, where the likelihood that
the RBM associates with a vector x is given by
pλ(X = x) =
X
h
pλ(X = x, H = h).
A popular approach to learn the RBM parameters is via gradient-based optimization, where
the gradients are approximated using contrastive divergence (Hinton et al., 2006; Bengio, 2009).
4
RBM in the Conditional Independence Case
In this section we show that given observed data x(1), . . . , x(n) ∈{0, 1}d from the condi-
tional independence model of Eq. (2), the posterior probabilities of the true, unknown labels
y(1), . . . , y(n) can be consistently estimated via a RBM with a single hidden node.
We begin by showing that there is a bijective map from the parameters λ of a RBM with a
single hidden node to the parameters θ of the conditional independence model, such that the
joint distribution speciﬁed by the RBM is equivalent to that of the conditional independence
model.
Lemma 4.1. The joint probability pλ(X = x, H = y) of a RBM with parameters λ = (a, b, W)
is equivalent to the joint probability pθ(X = x, Y = y) of a conditional independence model with
parameters θ = ({ψi}, {ηi}, π) given by
ψi ≡σ(ai + Wi), ηi ≡1 −σ(ai)
π ≡
P
x∈{0,1}d eaT x+b+xT W
P
x∈{0,1}d
 eaT x + eaT x+b+xT W 
Furthermore, the map λ 7→θ is a bijection.
We are now ready to prove the main result of this section, namely, that the posterior
distribution of the true labels y(1), . . . , y(n) can be consistently estimated by a RBM with a
single hidden node. To do so, we rely on a special case of a result proved by Chang (1996),
that provides conditions under which the parameters of the conditional independence model
are identiﬁable.
Lemma 4.2. Let x(1), ..., x(n) be observed data from the conditional independence model, spec-
iﬁed by pθ. Assume that θ is such that for each i = 1, . . . , d, Xi is not independent of Y (i.e.,
each classiﬁer is not just a random guess), and that d ≥3. Let ˆλMLE be a maximum likelihood
parameter estimate of a RBM with a single hidden node. Then the RBM posterior probability
pˆλMLE(H = 1|X = x) converges to the true posterior pθ(Y = 1|X = x), as n →∞.
5

## Page 6

Remark 4.3. The identiﬁability of the parameters is up to a single global 0/1 label ﬂip. This
means that one recovers either pθ(Y = y|X) or pθ(Y = 1−y|X). Assuming that on average, the
Xi’s are more accurate than a random guess, this sign ambiguity can be resolved by comparing
the predictions to the majority vote decision.
Remark 4.4. Lemma 4.2 assumes that we found the MLE of the RBM parameters. Obtaining
such a MLE is problematic for two main reasons. First, RBMs are typically trained to maximize
a proxy for the likelihood, as the true likelihood is not tractable. Second, the RBM likelihood
function is not concave, hence there are no guarantees that after training a RBM one obtains
the maximum likelihood parameter ˆλMLE.
5
RBM-based Deep Neural Net
In many practical settings, the variables X1, . . . , Xd are not conditionally independent. Fitting
a conditionally independent model to such data may yield highly sub-optimal predictions for
the true labels yi. To tackle this general case, we propose to train a RBM-based Deep Neural
Net (DNN) and use it to estimate the posterior probabilities pθ(Y |X). In such a DNN, the
hidden layer of each RBM is the input for the successive RBM. As suggested by Hinton et al.
(2006), the RBMs are trained one at a time, bottom to top, i.e., the DNN is trained in a
layer-wise fashion. Speciﬁcally, given training data x(1), . . . , x(n) ∈{0, 1}d, we start by training
the bottom RBM, and then obtain the ﬁrst layer hidden representation of the data by sampling
h(i) from the conditional RBM distribution pλ(H|X = x(i)). The vectors h(1), . . . , h(n) are then
used as a training set for the second RBM and so on.
In the case considered in this manuscript, where the true label y is binary, the upper-most
RBM in the DNN has a single hidden unit, from which the posterior probability pθ(Y |X) can
be estimated. Such a DNN is depicted in Figure 3.
5.1
Motivation
Deep learning algorithms have recently achieved state-of-the-art performance in a wide range
of applications LeCun et al. (2015). While a rigorous theoretical understanding of deep nets
is still lacking, many researchers believe that a key property in their success is their ability
to disentangle factors of variation in the inputs; see for example Bengio et al. (2013), Tishby
and Zaslavsky (2015), and Mehta and Schwab (2014). That is, as one moves through the net,
the hidden units become less statistically dependent. We have seen in Section 4 that given a
representation in which the units are independent conditional on the true label, a single node
RBM gives a consistent estimation of the true label posterior probability. Propagating the
data through several RBM layers can hence be seen as a processing of the data, which reduces
the conditional dependence of the units while preserving most of the information on the true
label Y . In Section 6 we will demonstrate cases where such decoupling does indeed happen
in practice, i.e., although the original input variables Xi’s are not conditionally independent
given the true label Y , after training, the units in the uppermost hidden layer are, remarkably,
6

## Page 7

ˆY
H2
1
H2
i
H2
m2
H1
1
H1
i
H1
m1
X1
Xi
Xd
Figure 3: A sketch of RBM-based DNN with two hidden layers.
approximately conditionally independent. Thus, the assumptions of the conditional indepen-
dence model apply (with respect to the uppermost hidden layer Hlast), and therefore one is
able to consistently estimate the label posterior probability, Pr(Y |Hlast), as in Section 4.
Another motivation for using deep nets with several hidden layers for unsupervised en-
semble learning is their rich expressive power. In our setting, we wish to approximate the
posterior probability p(Y |X), which in general may be a complicated nonlinear function of
X. When p(Y |X) cannot be accurately estimated by a RBM with a single hidden node (i.e.,
when the conditional independence assumption of Dawid and Skene does not hold), a better
approximation may be obtained from a deeper network. Several works show that there exist
functions that are signiﬁcantly more eﬃciently represented by deeper networks, compared to
shallower ones, where eﬃciency corresponds to the number of units. For example, Montufar
et al. (2014) show that deep networks with piece-wise linear activations can represent functions
with greater number of linear regions compared to shallow networks with the same number of
units. In a recent work, Eldan and Shamir (2015) give an example for a radial function that
can be eﬃciently computed by a 3-layer network, while requiring exponentially many units to
be approximated accurately by a 2-layer network.
Finally, we would like to emphasize that a RBM-based DNN is a discriminative model to
estimate the posterior p(Y |X). In general, it may not correspond to any generative model Arora
et al. (2015). Indeed, there is no guarantee that the marginal distributions implied by two
adjacent RBMs match.
Yet, it can be shown (see Appendix C) that stacking RBMs is a
variational inference procedure assuming a speciﬁc class of data generation models. The nature
of approximation of a top down generative model, where the data X is generated from a label
Y , by a RBM-based DNN is explored in Appendix D.
7

## Page 8

5.2
Predicting the Label from a Trained DNN
Given a trained DNN and a sample x ∼pθ(X), the label y is estimated by propagating x
through the network. Speciﬁcally, the units of each layer can be set by either (i) sampling from
the conditional distribution given the layer below, i.e., hj ∼pλ(hj|x), or (ii) by MAP estimate,
setting each hidden unit hj = arg maxhj∈{0,1} pλ(hj|x). Since the ﬁrst option is stochastic, one
may propagate x through the net multiple times and average the outputs p(y|x) to obtain an
approximation of E(Y |X = x). Experimentally, we found both options to be equally eﬀective,
while each option slightly outperforms the other in some cases.
5.3
Choosing the DNN Architecture
The speciﬁc DNN architecture (i.e., number and sizes of layers) might have a dramatic eﬀect
on the quality of predictions. To determine the number of units in each layer we employed
the following procedure: we ﬁrst train a RBM with d hidden units. Next, we compute the
singular value decomposition of the weight matrix W, and determine its rank (i.e., the number
of suﬃciently large singular values). Given that the rank is some m ≤d, we re-train the RBM,
setting the number of hidden units to be m. If m > 1, we add another layer on top of the
current layer, and proceed recursively. The process stops when m = 1, so that the last layer
of the DNN contains a single node. We refer to this method as the SVD approach. In our
experiments, as a rule of thumb, we set m to be the minimal number of singular values (in
descending order) whose cumulative sum is at least 95% of the total sum.
This method takes advantage of the co-adaptation of hidden units, which is a well known
phenomenon in RBM training (see, for example, Hinton et al. (2012)). The term co-adaptation
describes a situation where several hidden units tend to behave very similarly; this implies that
the rank of the weight matrix might be small, although the number of hidden units may be
larger.
6
Experimental Results
In this section we compare the performance of the proposed DNN approach to several other
approaches, and report experimental results obtained on four simulated data sets and eight real
world data sets, from two diﬀerent domains. All our datasets, as well as the scripts reproducing
the reported results are publicly available at https://github.com/ushaham/RBMpaper. 1.
Speciﬁcally, we compare between the following unsupervised ensemble methods:
• Vote. Majority voting, which is the maximum likelihood prediction, assuming that all
classiﬁers are conditionally independent and have the same accuracy.
• DS. Approximate maximum likelihood predictions under the Dawid and Skene model.
Speciﬁcally, we use Spectral Meta Learner (Parisi et al., 2014), and Restricted Likeli-
hood (Jaﬀe et al., 2014).
1 Our scripts are based on the publicly available code in Hinton’s website http://www.cs.toronto.edu/
~hinton/MatlabForSciencePaper.html.
8

## Page 9

• CUBAM The method of Welinder et al. (2010), which assumes conditional indepen-
dence, but allows the accuracy of each classiﬁer to vary across diﬀerent regions of the
input domain.
• L-SML Latent SML (Jaﬀe et al., 2015). This method relaxes the conditional indepen-
dence assumption to a depth 2 tree model.
• DNN The approach presented in this manuscript, with the depth and number of hidden
units in each layer determined by the SVD approach, described in Section 5.3.
Following Jaﬀe et al. (2015), the performance measure we chose is the balanced accuracy,
given by
P I{true label is 0 and predicted label is 0}
2 P I{true label is 0}
+
P I{true label is 1 and predicted label is 1}
2 P I{true label is 1}
,
where I{·} is the indicator function.
6.1
Simulated Datasets
In this experiment we carefully generated four synthetic datasets, in order to demonstrate the
performance of the DNN approach in several speciﬁc scenarios. In all four datasets the observed
data is a n × d binary matrix, with input dimension d = 15 and sample size n = 10, 000. A
detailed description of the datasets generation process is given in Appendix E.1.
• CondInd A dataset where the conditional independence holds, and 10 of the 15 classiﬁers
are in fact random guess.
• Tree15-3-1 A dataset generated from a depth-2 tree with layer sizes 1,3,15. Every node
in the intermediate layer is connected to ﬁve nodes in the bottom layer. This dataset
is generated from the model considered by L-SML, and does not satisfy the conditional
independence assumption, as is shown in Figure 6.
• LayeredGraph15-5-5-1 A dataset generated from a depth-3 layered graph, with layer
sizes 1,5,5,15.
In this case, the conditional independence assumption does not hold,
although in practice the amount of dependence in the data is not high (see Figure 11).
• TruncatedGaussian. Here X = (1 + sign(Z))/2, where the r.v. Z follows a a mixture
of two d-dimensional Gaussians with diﬀerent means and same covariance matrix. The
label Y indicates the speciﬁc Gaussian from which X is sampled. In this case, the data
is highly dependent, as can be seen in Figure 11.
The results are summarized in Table 1. Along with the ﬁve unsupervised methods, the table
also shows the accuracy of a supervised learner and the estimated accuracy of the Bayes-optimal
9

## Page 10

Table 1: Balanced accuracy of various unsupervised ensemble methods on the four synthetic
datasets, along with a supervised learner (SUP), and the Bayes optimal classiﬁer (Bayes-Opt).
The results are presented as mean ± standard deviation, based on 5 repetitions, where in each
repetition a new dataset was sampled from the model. The numbers in brackets denote the
architecture of the DNN, found by the SVD approach.
method
condInd
Tree15-3-1
LG15-5-5-1
TG
Vote
75.93 ± 0.5
93.45 ± 0.19
76.61 ± 0.09
80.14 ± 0.4
DS
94.78 ± 0.13
92.68 ± 0.14
86.36 ± 0.2
82.03 ± 0.27
CUBAM
91.96 ± 0.18
90.74 ± 0.3
77.12 ± 0.26
83.43 ± 0.31
L-SML
55.94 ± 21.88
95.83 ± 0.15
85.87 ± 0.21
79.5 ± 1.35
DNN
94.78 ± 0.13 (15-1)
95.13 ± 0.71 (15-3-1)
86.83 ± 0.2 (15-4-1)
88.09 ± 0.52 (15-3-1)
SUP
94.45 ± 0.11
95.54 ± 0.27
87.01 ± 0.18
90.8 ± 0.4
Bayes-Opt
95.32
96.12
87.05
91.39
Figure 4: The RBM weight vector on the condInd dataset. The hidden unit is strongly con-
nected only to the ﬁrst ﬁve visible units, reﬂecting the fact that in an unsupervised manner,
the RBM detected that the remaining units are random guess classiﬁers.
classiﬁer. The supervised learner is a Multi Layer Perceptron (MLP) with two hidden layers
of sizes 4 and 2, that was trained on a dataset with n = 10, 000 samples (independent of the
test dataset). The Bayes-optimal approximated accuracy was computed on a sample of size
10, 000, with the true posterior probabilities of all 2d possible binary vectors estimated using a
sample of size 106 from the corresponding model.
On all of the above datasets, the DNN always outperformed the majority vote rule and
CUBAM. On the CondInd dataset, the DNN performs similarly to DS, and signiﬁcantly better
than the other methods. Despite being unsupervised, on this dataset both methods perform
slightly better than the speciﬁc supervised learner we considered, and around the Bayes-optimal
accuracy. The architecture determined by the SVD approach in this case is indeed a single
RBM (with a single hidden node). The weight matrix of the RBM is shown in Figure 4, and
corresponds to the fact that only the ﬁrst ﬁve classiﬁers actually contain information about the
true label in this dataset.
Figure 5 shows the recovery of the true conditional independence model parameters {ψi, ηi}
of a similar conditional independent dataset (however with no random guess classiﬁers) from a
RBM with a single hidden node, using the map in Lemma 4.1.
10

## Page 11

Figure 5: Recovery of the conditional Independence model parameters {ψi, ηi} from a RBM
with a single hidden node, on a dataset sampled from a conditional independence model. The
parameters were uniformly sampled from [0.5, 1]. Each circle corresponds to a single parameter
(e.g., ψi for some i). For convenience, the identity line was added to the plot.
On the Tree15-3-1 dataset, L-SML, which is tailored for data generated by a tree, outper-
forms the DNN. This result is expected, since it can be shown that the distribution of the
bottom two layers of a tree cannot be parametrized as a RBM (see Appendix D). Still, the
DNN performs signiﬁcantly better than DS, CUBAM and majority vote, and not far from the
supervised learner and the optimal Bayes classiﬁer. Figure 6 shows the correlation matrix at
the input and hidden layers, as well as the ﬁrst layer weight matrix, demonstrating that the
DNN captured the true data generation model. Consequently, the 3 hidden units are nearly
conditionally uncorrelated given the label y.
Figure 7 shows the cumulative proportion of the singular values on the condInd and
Tree15-3-1 datasets, which explains the architecture determined by the SVD approach for
both datasets.
On the LayeredGraph15-5-5-1 dataset, while outperforming the other methods, the DNN
achieved accuracy close to the supervised learner and the Bayes optimal accuracy; however,
the chosen DNN architecture is diﬀerent from the one of the true data generation model.
The conditional independence assumption is strongly violated in the case of the Truncat-
edGaussian dataset. Here the DNN performs better than all other methods by a large margin.
6.2
Real-World Datasets
In this section we experiment with two groups of datasets, from two diﬀerent domains, as
follows:
• DREAM Three datasets from the DREAM mutation calling challenge Ewing et al. (2015);
this challenge is an international eﬀort to improve standard methods for identifying
cancer-associated mutations and rearrangements in whole-genome sequencing data. The
accuracy of current variant calling algorithms is not optimal due to sequencing errors,
other experimental factors, parametric choices in each algorithm and preprocessing and
11

## Page 12

Figure 6: The Tree15-3-1 experiment. Top left: correlation matrix of the input data for the
y = 0 class. The ﬁrst and middle ﬁve X′
is are not conditionally independent of each other.
Top right: correlation matrix of the hidden layer of the DNN for the y = 0 class. The hidden
units are approximately uncorrelated. Bottom: weight matrix of the bottom RBM of the DNN,
showing that each hidden unit is strongly connected to 5 visible units, as in the original data
generation model.
ﬁltering decisions. Unsupervised ensemble learning of multiple variant callers is expected
to provide more robust predictions. One of the goals of this challenge is to develop a
state-of-the-art meta pipeline for somatic mutation detection, to output accurate as pos-
sible mutation calls associated with cancer. Speciﬁcally, we used three datasets, (S1, S2,
S3) containing the predictions of classiﬁers that determine the presence or absence of of
mutations in genome sequencing data. The data is available at (Ellrot, 2013). In S1,
d = 124, n = 92, 362. In S2, d = 114, n = 70,561. In S3, d = 99, n = 78, 643.
• Magic Forty datasets, which are constructed from the Magic dataset in the UCI reposi-
tory, available at https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope.
This dataset contains n = 19, 020 instances with 11 attributes, which consists of physical
measurements of gamma particles; the learning task is to classify each instance as back-
ground or high energy gamma rays. Each of the ﬁve datasets we constructed contains
12

## Page 13

Figure 7: Cumulative proportion of singular values on the condInd and Tree15-3-1 datasets.
While in the condInd case the ﬁrst singular value is more than 95% of the total sum of singular
values, the ﬁrst three singular values are needed on the Tree15-3-1 dataset. The horizontal line
at 0.95 is added to the plot for convenience.
Dataset
Vote
DS
CUBAM
L-SML
DNN
S1
97.2 *
98.3 *
92.31
98.4 *
98.42 ± 0.0 (124-1)
S2
96 *
97.2 *
69.19
97.7 *
97.55 ± 0.01 (114-1)
S3
95.7 *
97.7 *
87.65
98.2 *
98.51 ± 0.01 (99-25-1)
Table 2: Balanced accuracy of various methods on the DREAM datasets S1, S2 and S3. DNN
results are averaged over 5 repetitions, and are presented as mean ± standard deviation. The
numbers in brackets denotes the architecture of the DNN, found by the SVD approach. *
results reported in (Jaﬀe et al., 2015)
binary predictions of d = 16 classiﬁers, obtained in the Weka machine learning software.
The 16 classiﬁers belong to four groups: four random forest classiﬁers, three logistic trees
classiﬁers, four SVM classiﬁers, and ﬁve naive Bayes classiﬁers. This setting is adopted
from Jaﬀe et al. (2015). The group of SVM classiﬁers is highly correlated, as well as
the group of Naive Bayes classiﬁers, as can be seen in Appendix E.2. Each of the forty
datasets was obtained by predictions of the same classiﬁers, however trained on a diﬀerent
subset of the original Magic dataset (a random subset of size 500 each time).
Table 2 shows the performance of the various methods on the DREAM datasets. As can be
seen, the DNN and L-SML performs similarly on S1, while the former performs better on S3
and the latter on S2. The two methods outperform the majority vote rule, DS and CUBAM
on all three datasets. Remarkably, the hidden representation on the S3 dataset is such that the
units are perfectly uncorrelated, conditioned on the hidden label. This is shown in Figure 8.
The results on the Magic datasets are shown in Figure 9. On most of these datasets, the
DNN outperforms all other methods, with a relatively large margin. On all forty datasets, the
SVD approach yielded a 15-3-1 architecture.
To summarize our experiments, we observed that RBM-based DNN performs at least as
well and often better than various other methods, on both simulated and real datasets, and that
the SVD approach can serve as an eﬀective tool for determination of the DNN architecture.
13

## Page 14

Figure 8: correlation matrices of the input (left) and hidden (right) layers of the DNN on the
S3 dataset, for the y = 0 class. Remarkably, the hidden units are almost perfectly uncorrelated,
conditioned on the class.
Figure 9: Performance of the various methods on the Magic datasets. For convenience, the
identity line is added to the plot. Most of the points are below the identity line, which indicates
that the DNN tend to outperform all other methods on these datasets.
We remark that in our experiments, we observed that RBMs tend to be highly sensi-
tive to hyper-parameter tuning (such as learning rate, momentum, regularization type and
penalty), and these hyper-parameters need to be carefully tuned.
To obtain a reasonable
hyper-parameter setting we found it useful to apply the random conﬁguration sampling pro-
cedure, proposed in (Bergstra and Bengio, 2012), and evaluate diﬀerent models by average
log-likelihood approximation, (see, for example, (Salakhutdinov and Murray, 2008) and the
corresponding MATLAB scripts in (Salakhutdinov, 2010)).
14

## Page 15

7
Summary and Discussion
We demonstrated how deep learning techniques can be used for unsupervised ensemble learning,
and showed that the DNN approach proposed in this manuscript often performs at least as well
and often better than state-of the art methods, especially when the conditional independence
assumption made by Dawid and Skene (1979) does not hold.
Possible directions for future research include extending the approach to multiclass prob-
lems, possible using Discrete RBMs Mont´ufar and Morton (2013), theoretical analysis of the
SVD approach, and information theoretic analysis of the de-correlation, while preserving label
information, that occurs while propagating data through a RBM-based DNN.
Acknowledgements
The authors would like to thank George Linderman, Alex Cloninger, Tingting Jiang, Raphy
Coifman, Sahand Negahban, Andrew Barron, Alex Kovner, Shahar Kovalsky, Maria Angelica
Cueto, Jason Morton, and Brend Strumfels for their help.
References
Arora, S., Liang, Y., and Ma, T. (2015). Why are deep nets reversible: A simple theory, with
implications for training. arXiv preprint arXiv:1511.05653.
Bengio, Y. (2009). Learning deep architectures for ai. Foundations and trends R⃝in Machine
Learning, 2(1):1–127.
Bengio, Y., Courville, A., and Vincent, P. (2013). Representation learning: A review and new
perspectives. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 35(8):1798–
1828.
Bergstra, J. and Bengio, Y. (2012). Random search for hyper-parameter optimization. The
Journal of Machine Learning Research, 13(1):281–305.
Bishop, C. M. (2006). Pattern recognition and machine learning. springer.
Blei, D. M., Kucukelbir, A., and McAuliﬀe, J. D. (2016). Variational inference: A review for
statisticians. arXiv preprint arXiv:1601.00670.
Casella, G. and Berger, R. L. (2002). Statistical inference, volume 2. Duxbury Paciﬁc Grove,
CA.
Chang, J. T. (1996). Full reconstruction of markov models on evolutionary trees: identiﬁability
and consistency. Mathematical biosciences, 137(1):51–73.
15

## Page 16

Cueto, M. A., Morton, J., and Sturmfels, B. (2010). Geometry of the restricted boltzmann
machine. Algebraic Methods in Statistics and Probability,(eds. M. Viana and H. Wynn),
AMS, Contemporary Mathematics, 516:135–153.
Dawid, A. P. and Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates
using the em algorithm. Applied statistics, pages 20–28.
Donmez, P., Lebanon, G., and Balasubramanian, K. (2010). Unsupervised supervised learning
i: Estimating classiﬁcation and regression errors without labels. The Journal of Machine
Learning Research, 11:1323–1351.
Eldan, R. and Shamir, O. (2015). The power of depth for feedforward neural networks. arXiv
preprint arXiv:1512.03965.
Ellrot, K. (2013). Icgc-tcga dream mutation calling challenge. https://www.synapse.org/#!
Synapse:syn312572/wiki/58893. Online; accessed 12-November-2015.
Ewing, A. D., Houlahan, K. E., Hu, Y., Ellrott, K., Caloian, C., Yamaguchi, T. N., Bare, J. C.,
P’ng, C., Waggott, D., Sabelnykova, V. Y., et al. (2015). Combining tumor genome simu-
lation with crowdsourcing to benchmark somatic single-nucleotide-variant detection. Nature
methods.
Fox, C. W. and Roberts, S. J. (2012). A tutorial on variational bayesian inference. Artiﬁcial
intelligence review, 38(2):85–95.
Hinton, G. E., Osindero, S., and Teh, Y.-W. (2006). A fast learning algorithm for deep belief
nets. Neural computation, 18(7):1527–1554.
Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. R. (2012).
Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint
arXiv:1207.0580.
Jaﬀe, A., Fetaya, E., Nadler, B., Jiang, T., and Kluger, Y. (2015). Unsupervised ensemble
learning with dependent classiﬁers. arXiv preprint arXiv:1510.05830.
Jaﬀe, A., Nadler, B., and Kluger, Y. (2014). Estimating the accuracies of multiple classiﬁers
without labeled data. arXiv preprint arXiv:1407.7644.
Jain, P. and Oh, S. (2013). Learning mixtures of discrete product distributions using spectral
decompositions. arXiv preprint arXiv:1311.2972.
LeCun, Y., Bengio, Y., and Hinton, G. (2015). Deep learning. Nature, 521(7553):436–444.
Mehta, P. and Schwab, D. J. (2014). An exact mapping between the variational renormalization
group and deep learning. arXiv preprint arXiv:1410.3831.
Mont´ufar, G. and Morton, J. (2013). Discrete restricted boltzmann machines. arXiv preprint
arXiv:1301.3529.
16

## Page 17

Montufar, G. F., Pascanu, R., Cho, K., and Bengio, Y. (2014).
On the number of linear
regions of deep neural networks. In Advances in Neural Information Processing Systems,
pages 2924–2932.
Parisi, F., Strino, F., Nadler, B., and Kluger, Y. (2014). Ranking and combining multiple pre-
dictors without labeled data. Proceedings of the National Academy of Sciences, 111(4):1253–
1258.
Platanios, A., Blum, A., and Mitchell, T. M. (2014). Estimating accuracy from unlabeled data.
In In Proceedings of UAI.
Raykar, V. C., Yu, S., Zhao, L. H., Valadez, G. H., Florin, C., Bogoni, L., and Moy, L. (2010).
Learning from crowds. The Journal of Machine Learning Research, 11:1297–1322.
Salakhutdinov, R. (2010). Ruslan salakhutdinov’s web page.
Salakhutdinov, R. and Murray, I. (2008). On the quantitative analysis of deep belief networks.
In Proceedings of the 25th international conference on Machine learning, pages 872–879.
ACM.
Tishby, N. and Zaslavsky, N. (2015). Deep learning and the information bottleneck principle.
arXiv preprint arXiv:1503.02406.
Welinder, P., Branson, S., Perona, P., and Belongie, S. J. (2010). The multidimensional wisdom
of crowds. In Advances in neural information processing systems, pages 2424–2432.
Whitehill, J., Wu, T.-f., Bergsma, J., Movellan, J. R., and Ruvolo, P. L. (2009). Whose vote
should count more: Optimal integration of labels from labelers of unknown expertise. In
Advances in neural information processing systems, pages 2035–2043.
Zhang, Y., Chen, X., Zhou, D., and Jordan, M. I. (2014).
Spectral methods meet em: A
provably optimal algorithm for crowdsourcing. In Advances in neural information processing
systems, pages 1260–1268.
17

## Page 18

A
Proof of Lemma 4.1
Proof. We will deﬁne θ so that for every x, y, pθ(Xi = xi|Y = y) = pλ(Xi = xi|H = y) and
pθ(Y = y) = pλ(H = y).
Since the weight matrix W has dimension d × 1 in this case, it is a vector, which we will
denote as w. Recall that
pλ(Xi = 1|H = y) = σ(ai + wiy),
hence we deﬁne
ψi ≡σ(ai + wi)
and
ηi ≡1 −σ(ai).
Finally, recall that
pλ(H = 1) =
P
x∈{0,1}d e−Eλ(x,1)
P
x∈{0,1}d, h∈{0,1} e−Eλ(x,h)
=
P
x∈{0,1}d eaT x+b+xT w
P
x∈{0,1}d, eaT x + eaT x+b+xT w ,
where Eλ is the energy function given in equation (3), hence we set
π ≡
P
x∈{0,1}d eaT x+b+xT w
P
x∈{0,1}d,
 eaT x + eaT x+b+xT w.
(4)
To see that the map λ 7→θ is 1:1, note that ai uniquely determines ηi, hence (ai, wi) uniquely
determine (ψi, ηi). Lastly, rearranging equation (4) we get
π
X
x∈{0,1}d

eaT x + eaT x+b+wT x
=
X
x∈{0,1}d
eaT x+b+wT x
⇒π
X
x∈{0,1}d
eaT x = (1 −π)eb
X
x∈{0,1}d
eaT x+wT x
⇒eb =
π
1 −π
P
x∈{0,1}d eaT x
P
x∈{0,1}d eaT x+wT x ,
so that given (a, W), π is uniquely determined by b. Showing that the map λ 7→θ is a also
subjective is straightforward. Hence it is a bijection.
B
Proof of Lemma 4.2
Proof. Since d ≥3 and for each i, Xi is not independent of Y , by Chang (1996), the parameter
θ of the conditional independence model is identiﬁable. Since the map λ 7→θ in Lemma 4.1
18

## Page 19

is a bijection, there exists λ corresponding to θ, which is therefore identiﬁable as well. By the
consistency property of the MLE (see, for example, (Casella and Berger, 2002)),
lim
n→∞
ˆλMLE = λ.
Since pλ(H = 1|X) is continuous in θ, one obtains
pˆλMLE(H = 1|X) →pλ(H = 1|X).
Finally, note that Lemma 4.1 implies, in particular, that under the map λ 7→θ
pλ(H = 1|X) = pθ(Y = 1|X),
which completes the proof.
C
Stacking RBMs as a Variational Inference Procedure
Variational inference is a common approach to tackle complicated probability estimation prob-
lems (see, for example, Bishop (2006); Fox and Roberts (2012), and a recent review Blei et al.
(2016)). Speciﬁcally, let p be a target probability distribution that we want to approximate.
In variational inference we deﬁne a family of approximate distributions D = {qα : α ∈A}, and
then perform optimization to ﬁnd the member of D that is closest to p in Kullback-Leibler
distance. A key idea is that the family D is ﬂexible enough to contain a distribution close to p,
yet simple enough to perform optimization over. For example, a popular choice is to take D as
the collection of factorized distributions, i.e., of the form qα(X) = Q
i qα(Xi). In this section,
we motivate the use of RBM-based DNN by considering a speciﬁc data generation model, and
showing that training a stack of RBMs on data generated by this model is in fact a variational
inference procedure.
The generative model we consider is a two layer Deep Belief Network (DBN), which played
an important role in the emergence of deep learning in 2006 Hinton et al. (2006). The DBN we
consider generates data Y ∈{0, 1}, H ∈{0, 1}m, X ∈{0, 1}d via the probability distribution
pθ(X, H, Y ) ≡pθ1(X, H)pθ2(Y |H)
where X, H form a RBM (parametrized by θ1).
We observe data x(1) . . . x(n) from pθ(X) and our goal is to estimate the posterior pθ(y(i)|x(i))
for i = 1, . . . n. The posterior can be written as
pθ(Y |X) = Eh∼pθ1(H|X)Pθ2(Y |H = h).
Cueto et al. (2010) showed that as long as m is not too large comparing to d, RBMs are
locally identiﬁable, i.e., identiﬁable up to order and ﬂips of hidden units (Jason Morton, personal
communication). Therefore, when training a RBM with m hidden units on x(1) . . . x(n), by the
consistency property of the MLE Casella and Berger (2002) the MLE ˆθ1MLE will converge to
19

## Page 20

the true parameter θ1 as n →∞. Hence, when n is large enough, the vectors h(i) obtained
from the (trained) RBM are in fact samples from pθ1(H|X = x(i)).
At the next step, the vectors h(1) . . . h(n) are used to train a second RBM, with a single
hidden node. Observe that in the data generation model considered in this section, pθ(H|Y )
does not factorize. The factorized distribution pλ(H|Y ) that minimizes KL(pθ2(H|Y )∥pλ(H|Y ))
is given by
pλ(Hi|Y ) = pθ2(Hi|Y )
Bishop (2006) (Chapter 10). By Lemma 4.1, we know that the distribution
pλ(H, Y ) = pθ(Y )
Y
i
pθ2(Hi|Y )
(5)
is equivalent to a RBM. Finally, by Lemma 4.2, the distribution (5) is consistently estimated
by a RBM trained on vectors h(1) . . . h(n), and is thus a variational inference procedure.
D
Stacking RBMs as an Approximation for a Directed Top-
Down Model
Assume that the data is generated by a Markov chain Y →H →X, where Y ∈{0, 1},
H ∈{0, 1}m, X ∈{0, 1}d.
We further assume that the distributions pθ(X|H), pθ(H|Y )
factorize, i.e.,
pθ(X|H) =
d
Y
i=1
Pr(Xi|H)
(6)
and
pθ(H|Y ) =
m
Y
i=1
Pr(Hi|Y ),
(7)
and are given by RBM-like conditional distributions, i.e.,
pθ(Xi = 1|H) = σ (ai + Wi,·H)
(8)
and
pθ(Hi = 1|Y ) = σ (bi + Ui,·Y ) .
(9)
Hence the corresponding data generation probability is parametrized by θ = (π, a, b, W, U),
where π = Pr(Y = 1).
This data generation process is depicted in Figure 10.
The posterior probabilities pθ(Y |X) are given by
pθ(Y |X) =
X
H∈{0,1}m
pθ(Y |H)pθ(H|X)
= Eh∼pθ(H|X)pθ(Y |H = h).
20

## Page 21

Y
H1
Hi
Hm
X1
Xi
Xd
Figure 10: Data generated by a Markov Chain Y →H →X with RBM-like conditional
probabilities.
By Section 4, we know that pθ(H, Y ) is equivalent to a RBM. Therefore, to accurately estimate
the posterior, it suﬃces to approximate pθ(H|X).
Under the data generation model described in Figure 10 and equations (6)-(9), it is evident
that the joint distribution pθ(X, H) cannot be parametrized as a RBM; indeed, pθ(H|X) does
not factorize. Hence, training a RBM on samples from pθ(X), is a mean ﬁeld approximation
of pθ(H|X). The form of pθ(X, H) is shown in the following lemma.
Lemma D.1. Under the data generation model described in Figure 10 and equations (6)-(9),
the joint distribution pθ(X, H) is given by
pθ(X, H) = exp
 aT X + XT WH + bT H

Z(H)
where
Z(H) =
1
P
X∈{0,1}d exp (aT X + XT WH)
×
X
Y ∈{0,1}
pθ(Y ) exp(HT UY )
P
H′ exp (bT H′ + H′T UY )
Proof. By deﬁnition,
pθ(X, H) =
X
Y ∈{0,1}
pθ(X, H, Y )
=
X
Y ∈{0,1}
p(Y )pθ(H|Y )p(X|H)
(10)
Writing
pθ(X|H) =
exp
 aT X + XT WH

P
X′∈{0,1}d exp (aT X′ + X′T WH)
21

## Page 22

and similarly
pθ(H|Y ) =
exp
 bT H + HT UY

P
H′∈{0,1}m exp (bT H′ + H′T UY ),
we obtain
pθ(X|H)pθ(H|Y ) =
exp
 aT X + XT WH + bT H + HT UY

(P
X′ exp (aTX′ + X′T WH)) (P
H′ exp (bT H′ + H′T UY )).
(11)
Plugging equation (11) in equation (10) we get
pθ(X, H) = exp
 aT X + XT WH + bT H

×
1
P
X′ exp (aTX′ + X′T WH)
×
X
Y ∈{0,1}
pθ(Y ) exp(HT UY )
P
H′ exp (bT H′ + H′T UY )
From lemma D.1 we see that pθ(H|X) is close to be factorizable if Z(H) is a approximately
a log-linear function of H and pθ(X) is approximately a log-linear function of X.
E
Datasets used for our experiments
E.1
Simulated Dataset Generation Details
• CondInd: the label Y was sampled from a Bernoulli(0.5) distribution; The speciﬁcity
ηi and sensitivity ψi of the variables Xi, i = 1 . . . 5 were sampled uniformly from [0.5, 1].
The other ten Xi’s were random guesses, i.e., had speciﬁcity = sensitivity = 0.5.
• Tree15-3-1: the label Y was sampled from a Bernoulli(0.5) distribution; each node in
the intermediate and layer was generated from his parent with speciﬁcity and sensitivity
sampled uniformly from [0.8, 1], and in the bottom layer with speciﬁcity and sensitivity
sampled uniformly from [0.6, 1].
• LayeredGraph15-5-5-1: Data is generated from a Layered Graph with four layers of
dimensions 1,5,5,15, starting at the true label Y . Each layer in the graph is generated from
the above layer, and the graph has sparse connectivity (about 30% of the edges exist).
For every node i and parent j we sample speciﬁcity ψij and sensitivity ηij uniformly.
Finally, the value at each node was calculated as the weighted sum of the probabilities of
the node being 1 given the values of the nodes in the preceding layer, normalized by the
sum over the edges. The label Y was sampled from a Bernoulli(0.5) distribution.
22

## Page 23

• TruncatedGaussian: the label Y was sampled from a Bernoulli(0.5) distribution. One
Gaussian had mean vector µ1 were each of the 15 coordinates was sampled uniformly.
The other Gaussian had mean vector µ2 = −µ1. Both Gaussians had identical covariance
matrix, with oﬀdiagonal entries of 0.5 and diagonal entries of 1.
Figure 11: correlation matrices of the input data, for the y = 0 class in all four simulated
datasets: condInd (top left), tree15-3-1 (top right), LayeredGraph (bottom left), Truncat-
edGaussian (bottom right).
E.2
The Magic Datasets
An example for the correlation matrix of the 16 classiﬁers given the 0 class can be seen in
Figure 12.
23

## Page 24

Figure 12: correlation matrix of the 16 classiﬁers in the Magic1 dataset, for the y = 0 class.
24
