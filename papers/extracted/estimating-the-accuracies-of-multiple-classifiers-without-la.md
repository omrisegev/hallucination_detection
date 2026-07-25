---
source_pdf: papers/Estimating the Accuracies of Multiple Classifiers Without Labeled Data.pdf
slug: estimating-the-accuracies-of-multiple-classifiers-without-la
pages: 27
extracted_on: 2026-07-13
---

# Estimating the Accuracies of Multiple Classifiers Without Labeled Data

## Page 1

arXiv:1407.7644v2  [stat.ML]  30 Oct 2014
Estimating the Accuracies of Multiple Classiﬁers
Without Labeled Data
Ariel Jaﬀe1,*, Boaz Nadler1,**, and Yuval Kluger2,3,***
1Dept. of Computer Science and Applied Mathematics, Weizmann Institute of Science,
Rehovot Israel 76100
*ariel.jaffe@weizmann.ac.il
**boaz.nadler@weizmann.ac.il
2Dept. of Pathology, Yale University, School of Medicine, New Haven, CT 06520
3NYU Center for Health Informatics and Bioinformatics New York University, Langone
Medical Center, 227 East 3030th Street, New York, NY 10016, USA
***yuval.kluger@yale.edu
Abstract
In various situations one is given only the predictions of multiple clas-
siﬁers over a large unlabeled test data. This scenario raises the following
questions: Without any labeled data and without any a-priori knowledge
about the reliability of these diﬀerent classiﬁers, is it possible to consis-
tently and computationally eﬃciently estimate their accuracies? Further-
more, also in a completely unsupervised manner, can one construct a more
accurate unsupervised ensemble classiﬁer? In this paper, focusing on the
binary case, we present simple, computationally eﬃcient algorithms to
solve these questions.
Furthermore, under standard classiﬁer indepen-
dence assumptions, we prove our methods are consistent and study their
asymptotic error. Our approach is spectral, based on the fact that the
oﬀ-diagonal entries of the classiﬁers’ covariance matrix and 3-d tensor are
rank-one. We illustrate the competitive performance of our algorithms
via extensive experiments on both artiﬁcial and real datasets.
1
Introduction
Consider a classiﬁcation problem from an instance space X to an output label
set Y = {1, . . . , K}. In contrast to the classical supervised setting, in various
contemporary applications, one has access only to the predictions of multiple
experts or classiﬁers over a large number of unlabeled instances. Moreover, the
reliability of these experts may be unknown, and at test time there is no labeled
data to assess it. This occurs for example when due to privacy considerations
each classiﬁer is trained with its own possibly proprietary labeled data, un-
available to us. Another scenario is crowdsourcing, where an annotation task
1

## Page 2

over many instances is distributed to many annotators whose reliability is a-
priori unknown, see for example Welinder et al. [2010], Whitehill et al. [2009],
Sheshadri and Lease [2013].
This setup, denoted as unsupervised-supervised
learning in ?, appears in several other application domains, including decision
science, economics and medicine, see Snow et al. [2008], Raykar et al. [2010],
Parisi et al. [2014].
Given only the m × n matrix Z, or a signiﬁcant part of it, with Zij = fi(xj)
holding the predictions of the given m classiﬁers over n instances, and without
any labeled data, two fundamental questions arise: (i) Under the assumption
that diﬀerent classiﬁers make independent errors, is it possible to consistently
estimate the accuracies of the m classiﬁers in a computationally eﬃcient way;
and (ii) is it possible to construct, again by some computationally eﬃcient pro-
cedure, an unsupervised ensemble learner, more accurate than most if not all of
the original m classiﬁers.
The ﬁrst question is important in cases where obtaining the predictions of
these m classiﬁers is by itself an expensive task, and after collecting a certain
number of instances and their predictions, we wish to pick only a few of the
most accurate ones, see Rokach [2009]. The second question, also known as
oﬄine consensus, is of utmost importance in improving the quality of automatic
decision making systems based on multiple sources of information.
Beyond the simplest approach of majority voting, perhaps the ﬁrst to deﬁne
and address these questions were Dawid and Skene [1979]. With the increasing
popularity of crowdsourcing and large scale expert opinion systems, the last
years have seen a surge of interest in these problems, see Sheng et al. [2008],
Whitehill et al. [2009], Raykar et al. [2010], Platanios et al. [2014] and refer-
ences therein.
Yet, the most common methods to address questions (i) and
(ii) above are based on the expectation maximization (EM) algorithm, already
proposed in this context by Dawid and Skene, and whose only guarantee is
convergence to a local maxima.
Two recent exceptions, proposing spectral (and thus computationally ef-
ﬁcient) methods with strong consistency guarantees are Karger et al. [2011]
and Parisi et al. [2014]. Karger et al. [2011] assume a spammer-hammer model,
where each classiﬁer is either perfectly correct or totally random and develop a
spectral method to detect which one is which. Parisi et al. [2014] derive a spec-
tral approach to address questions (i) and (ii) above in the context of binary
classiﬁcation. Their approach, however, has several limitations. First, they do
not actually estimate each classiﬁer sensitivity and speciﬁcity, but only show
how to consistently rank them according to their balanced accuracies. Second,
their unsupervised learner assumes that all classiﬁers have balanced accuracies
close to 1/2 (random). Hence, their ensemble learner may be suboptimal, for
example, when few classiﬁers are signiﬁcantly more accurate than all others.
In this paper we extend and generalize the results of Parisi et al. [2014] in
several directions and make the following contributions: In Sec. 3, focusing on
the binary case, we present a simple spectral method to estimate the sensitivity
and speciﬁcity of each classiﬁer, assuming the class imbalance is known. Hence,
the problem boils down to estimating a single-scalar – the class imbalance. In
2

## Page 3

Section 4 we present two diﬀerent methods to do so. First, in Sec. 4.1, we
prove that the oﬀ-diagonal elements of the m × m covariance matrix and the
m × m × m joint covariance tensor of the set of classiﬁers are both rank 1.
Moreover the covariance matrix and tensor share the same eigenvector but with
diﬀerent eigenvalues, from which the class imbalance can be extracted by a
simple least-squares procedure. In Sec. 4.2, we devise a second algorithm to
estimate the class imbalance by a restricted likelihood approach. The maxima
of this function is attained at the class imbalance, and can thus be found by a
one-dimensional scan. Both algorithms are computationally eﬃcient, and under
the assumption that classiﬁers make independent errors, are also proven to be
consistent. For the ﬁrst method, we also prove it is rate optimal with asymptotic
error OP (1/√n), where n is the number of unlabeled samples. Our work thus
provides a simple and elegant solution to the long-standing problem originally
posed by Dawid and Skene [2], whose previous solutions were mostly based on
expectation maximization approaches to the full likelihood function.
In Sec. 5 we consider the multiclass case. Building upon standard reductions
from multiclass to binary, we devise a method to estimate the class probabilities
and the diagonal entries of the confusion matrices of all classiﬁers. We also
prove that in the multiclass case, using only the ﬁrst and second moments of
these binary reductions, it is in general not possible to estimate all entries of the
confusion matrices of all classiﬁers. This motivates the development of tensor or
higher order methods to solve the multi-class case, as for example in Zhang et al.
[2014]. In Sec. 6 we illustrate our methods on both real and artiﬁcial data.
The results on real data show that our proposed ensemble learner achieves a
competitive performance even in practical scenarios where the assumption of
independent classiﬁers’ errors does not hold precisely.
Related Work
Under the assumption that all classiﬁers make independent
errors, the crowdsourcing problem we address is equivalent to learning a mix-
ture of discrete product distributions. This problem was studied, among oth-
ers, by Freund and Mansour [1999] for the case of k = 2 distributions, and by
Feldman et al. [2008] for k > 2. Important observations regarding the low-rank
spectral structure of the second and third moments of such distributions were
made by Anandkumar et al. [2012a,b]. Building upon these results, recently
Jain and Oh [2013] and Zhang et al. [2014], devised computationally eﬃcient
algorithms to estimate the parameters of the mixture of product distributions,
which are equivalent to the confusion matrices and class probabilities in our
problem.
Our ﬁrst method to estimate the class imbalance in the binary case using the
mean-centered 3-d tensor is closely related to these works, with some notable
diﬀerences.
One key diﬀerence is that the above works study non-centered
tensors of classiﬁers’ outputs, and hence for a k-class problem, need to resolve
the structure of rank-k tensors. In contrast, we work with centered matrices and
tensors. In the binary case with k = 2, we thus obtain a simpler rank-1 tensor,
which we do not even need to decompose, but only extract a single scalar from
3

## Page 4

it. A second diﬀerence is that the above methods require stronger assumptions
on the classiﬁers. For example, Zhang et al. [2014] divide the classiﬁers into
groups and assume that within each group, on average classiﬁers are better
than random. Due to these diﬀerences, our resulting algorithm is signiﬁcantly
simpler.
Our second algorithm for estimating the class imbalance, based on a re-
stricted likelihood approach is totally diﬀerent from these tensor-based works,
as it requires only a spectral decomposition of the classiﬁers’ covariance matrix,
and then optimizes a 1-d function of the full likelihood of the data. On both
simulated and real data, this second approach had at least as good as, and in
some cases better accuracy compared to the tensor based method. Finally, while
we focus on classiﬁcation, our algorithms may also be of interest to learning a
mixture of discrete product distributions.
2
Problem Setup
We consider the following binary classiﬁcation problem, as also studied in several
works (Dawid and Skene [1979], Raykar et al. [2010], Parisi et al. [2014]). Let
X be an instance space with an output space Y = {−1, 1}. A labeled instance
(x, y) ∈X × Y is a realization of the random variable (X, Y ), which has an
unknown probability density p(x, y), and X and Y marginals pX(x) and pY (y),
respectively. We further denote by b the class imbalance of Y ,
b = Pr(Y = 1) −Pr(Y = −1) = pY (1) −pY (−1).
Let {fi}m
i=1 be m ≥3 classiﬁers operating on X. In this binary setting, the
accuracy of the i-th classiﬁer is fully speciﬁed by its sensitivity ψi and speciﬁcity
ηi,
ψi =
Pr (fi(X) = 1|Y = 1)
ηi =
Pr (fi(X) = −1|Y = −1) .
For future use, we denote by πi its balanced accuracy,
πi = (ψi + ηi)/2.
In this paper we consider the following totally unsupervised scenario. Let Z be a
m × n matrix with entries Zij = fi(xj), i = 1, . . . , m, j = 1, . . . , n, where fi(xj)
is the label predicted at instance xj by classiﬁer fi. In particular, we assume
no prior knowledge about the m classiﬁers, so their accuracies (sensitivities ψi
and speciﬁcities ηi) are all unknown.
Given only the matrix Z of binary predictions1, we consider the following two
problems: (i) consistently and computationally eﬃciently estimate the sensitiv-
ity and speciﬁcity of each classiﬁer, and (ii) construct a more accurate ensemble
1For simplicity of exposition, we assume the matrix is fully observed. While beyond the
scope of this paper, our proposed methods and theory continue to hold if few entries are
missing (at random), such that accurate estimates of various means, covariances and tensors,
as detailed in Sections 3-4 are still possible.
4

## Page 5

classiﬁer. As discussed below, under certain assumptions, a solution to the ﬁrst
problem readily yields a solution to the second one.
To tackle these problems, we make the following three assumptions: (i) The
n instances xj are i.i.d.
realizations from the marginal pX(x).
(ii) The m
classiﬁers are conditionally independent. That is, for every pair of classiﬁers
fi, fj with i ̸= j and for all labels ai, aj ∈{−1, 1},
Pr(fi = ai, fj = aj|Y = y) =
Pr(fi = ai|Y = y) Pr(fj = aj|Y = y).
(1)
(iii) Most of the classiﬁers are better than random, in the sense that for more
than half of all classiﬁers, πi > 0.5. Note that (i)-(ii) are standard assump-
tions in both the supervised and unsupervised settings, see Dietterich [2000],
Dawid and Skene [1979], Raykar et al. [2010], Parisi et al. [2014]. Assumption
(iii) or a variant thereof is needed, given an inherent ±1 sign ambiguity in this
fully unsupervised problem.
3
Estimating ψ and η with a known class imbal-
ance.
For some classiﬁcation problems, the class imbalance b is known. One example
is in epidemiology, where the overall prevalence of a certain disease in the pop-
ulation is known, and the classiﬁcation problem is to predict its presence, or
future onset, in individuals given their observed features (such as blood results,
height, weight, age, genetic proﬁle, etc).
Assuming b is known, ? presented a simple method to estimate the error
rates of all classiﬁers under a symmetric noise model, where ψi = ηi for all
i, and EM methods in the general case, see also Raykar et al. [2010]. We in-
stead build upon the spectral approach in Parisi et al. [2014], and present a
computationally eﬃcient method to consistently estimate the sensitivities and
speciﬁcities of all m classiﬁers. To motivate our approach, it is instructive to
study the limit of an inﬁnite unlabeled set size, n →∞, where the mean values
of the classiﬁers µi = E[fi(X)], and their m × m population covariance matrix
R = E [(fi(x) −µi)(fj(x) −µj)], are all perfectly known.
The following two lemmas show that R and {µi}m
i=1 contain the information
needed to extract the speciﬁcities and sensitivities of the m classiﬁers. Lemma
1 appeared in Parisi et al. [2014], and implies that given the value of b one
may compute the balanced accuracies of all classiﬁers. Lemma 2 is new and
shows how to extract their sensitivities and speciﬁcities. Its proof appears in
the appendix.
Lemma 1. The oﬀdiagonal elements of the matrix R are identical to those of
a rank one matrix vvT , whose vector v, up to a ±1 sign ambiguity, is equal to
v =
p
1 −b2(2π −1),
(2)
5

## Page 6

where the vector π = (π1, . . . , πm) contains the balanced accuracies of the m
classiﬁers.
Lemma 2. Given the class imbalance b, the vector µ = (µ1, . . . , µm) containing
the mean values of the m classiﬁers, and v of Eq.
(2), the values of ψ =
(ψ1, . . . , ψm) and η = (η1, . . . , ηm) with the speciﬁcities and sensitivities of the
m classiﬁers are given by
ψ = 1
2

1 + µ + v
q
1−b
1+b

, η = 1
2

1 −µ + v
q
1+b
1−b

.
(3)
To uniquely recover v from the oﬀ-diagonal entries of R, we further assume
that at least three classiﬁers have diﬀerent balanced accuracies, which are all
diﬀerent from 1/2 (so 2πi −1 ̸= 0). In practice, the quantities {µi}m
i=1, R and
consequently the eigenvector v are all unknown. We thus estimate them from
the given data, and plug into Eq. (3). Let us denote by ˆµ and ˆR the sample
mean and covariance matrix of all classiﬁers, whose entries are given by
ˆµi
=
1
n
n
X
k=1
fi(xk),
(4)
ˆrij
=
1
n −1
n
X
k=1
(fi(xk) −ˆµi)(fj(xk) −ˆµj).
Estimating the vector v from the noisy matrix ˆR can be cast as a low-rank
matrix completion problem. Parisi et al. [2014] present several methods to con-
struct such an estimate ˆv, and resolve its inherent ±1 sign ambiguity, via as-
sumption (iii). Inserting ˆµ and ˆv into (3), gives the following estimates for ψ
and η,
ˆψ = 1
2

1 + ˆµ + ˆv
q
1−b
1+b

, ˆη = 1
2

1 −ˆµ + ˆv
q
1+b
1−b

.
(5)
The following lemma, proven in the appendix, presents some statistical prop-
erties of ˆψ and ˆη.
Lemma 3. Under assumptions (i)-(iii) of Section 2, ˆψ and ˆη are consistent
estimators of ψ and η. Furthermore, as n →∞,
ˆψi = ψi + OP
 1
√n

,
ˆηi = ηi + OP
 1
√n

.
(6)
In summary, assuming the class imbalance b is known, Eq.
(5) gives a
computationally eﬃcient way to estimate the sensitivities and speciﬁcities of all
classiﬁers. Lemma 3 ensures that this approach is also consistent. In the next
section we show that the assumption of explicit knowledge of b can be removed,
whereas in Section 5 we show that a similar approach can also (partly) handle
the multiclass case.
6

## Page 7

3.1
Unsupervised Ensemble Learning
We now consider the second problem discussed in Section 2, the construction
of an unsupervised ensemble learner. To this end, note that under the stronger
assumption that all classiﬁers make independent errors, the likelihood of a label
y at an instance x with predicted labels f1(x), . . . , fm(x) is
L(f1(x), . . . , fm(x)) | y) =
m
Y
i=1
Pr(fi(x) | y).
(7)
In Eq. (7), the i-th term Pr(fi(x)|y) depends on the speciﬁcity and sensitivity
ψi and ηi of the i-th classiﬁer. While the likelihood is non-convex in ψi, ηi and
y, if the former are known, there is a closed form solution for the maximum-
likelihood value of the class label,
ˆy(ML) = sign (P
i fi(x) ln αi + ln βi)
(8)
where
αi =
ψiηi
(1 −ψi)(1 −ηi),
βi = ψi(1 −ψi)
ηi(1 −ηi) .
(9)
Parisi et al. [2014], assumed all classiﬁers are close to random, and via a Taylor
expansion near ψ = η = 1/2, showed that β is approximately zero, and αi ≈
1 + 4(2πi −1). Plugging these into Eq. (8), they derived the following spectral
meta-learner (SML),
ˆy(SML) = sign (P
i fi(x)ˆvi) .
(10)
Their motivation was that they only had estimates of the vector v, which ac-
cording to Eq. (2) is proportional to (2π−1). Since we consistently estimate the
individual speciﬁcities and sensitivities of the m classiﬁers, we suggest to plug in
these estimates directly into Eqs. (9) and (8). Our improved spectral approach,
denoted i-SML, yields a more accurate ensemble learner when few classiﬁers are
signiﬁcantly better than random, so the linearization around ψ = η = 1/2 is
inaccurate. We present such examples in Sec. 6. Finally, we note that as in
Parisi et al. [2014] and Zhang et al. [2014], we may use our i-SML as a starting
guess for EM methods that maximize the full likelihood.
4
Estimation of the class imbalance
We now consider the problem of estimating ψ and η when the class imbalance
b is unknown. Our proposed approach is to ﬁrst estimate b, and then plug this
estimate into Eq. (5). We present two diﬀerent methods to estimate the class
imbalance. The ﬁrst uses the covariance matrix and the 3-dimensional covari-
ance tensor of all m classiﬁers. The second method exploits properties of the
likelihood function. As detailed below, both methods are computationally eﬃ-
cient, but require stronger assumptions than Eq.(1) on independence of classiﬁer
errors to prove their consistency.
7

## Page 8

4.1
Estimation via the 3-D covariance tensor
For the method derived in this subsection, we assume that the classiﬁers are
conditionally independent in triplets. That is, for every fi, fj, fk with i ̸= j ̸= k
and for all labels ai, aj, ak ∈{−1, 1},
Pr(fi = ai, fj = aj, fk = ak|y) =
Pr(fi = ai|y) Pr(fj = aj|y) Pr(fk = ak|y).
(11)
Let T = (Tijk) denote the 3-dimensional covariance tensor of the m classiﬁers
{fi(X)}m
i=1,
Tijk = E [(fi(X) −µi)(fj(X) −µj)(fk(X) −µk)] .
(12)
The following lemma, proven in the appendix, provides the relation between the
tensor T , the class imbalance b and the balanced accuracies of the m classiﬁers.
Lemma 4. Under assumption (11), the following holds for all i ̸= j ̸= k,
Tijk = −2b(1 −b2)(2πi −1)(2πj −1)(2πk −1).
(13)
According to (13), the oﬀdiagonal elements of T (with i ̸= j ̸= k) correspond
to a rank one tensor,
T = w ⊗w ⊗w,
(14)
where ⊗denotes the outer product and the vector w ∈Rm is equal to
w =
 −2b(1 −b2)
 1
3 · (2π −1).
(15)
Note that unlike the vector v of the covariance matrix R, there is no sign
ambiguity in the vector w.
Moreover, comparing Eqs. (2) and (15), the vectors v of R and w of T are
both proportional to (2π −1), where the proportionality factor depends on the
class imbalance b. Hence, w = α(b)1/3 v, and
T = α(b) v ⊗v ⊗v
(16)
where α(b) = (−2b)/
√
1 −b2. Inverting this expression yields the following re-
lation,
b = −α/
p
4 + α2.
(17)
Eq. (17) thus shows, that in our setup, as n →∞, the ﬁrst three moments of
the data (µ, R, T ) are suﬃcient to determine both the class imbalance and the
sensitivities and speciﬁcities of all m classiﬁers.
In practice, the tensor T is unknown, though it can be estimated from the
observed data by
ˆTijk = 1
n
n
X
l=1
(fi(xl) −ˆµi)(fj(xl) −ˆµj)(fk(xl) −ˆµk).
(18)
8

## Page 9

Algorithm 1 Estimating class imbalance with the 3dimensional covariance
tensor
1: Estimate covariance matrix R by Eq. (5).
2: Estimate v from the oﬀdiagonal entries of ˆR (see appendix).
3: Estimate the 3 dimensional tensor T by Eq. (18).
4: Estimate α via Eq. (19) and b via Eq. (17).
Given an estimate ˆv from the matrix ˆR, the scalar α of Eq. (16) is estimated
by least squares,
ˆα = argmin
α
X
i<j<k

ˆTijk −α ˆviˆvjˆvk
2
.
(19)
A summary of the steps to estimate the class imbalance with the 3 dimensional
tensor appears in Algorithm 1. The following lemma shows that this method
yields an asymptotic error of OP (1/√n). This error rate is optimal since even
if we knew the ground truth labels yi, estimating b from them would still incur
such an error rate.
Lemma 5. Let ˆα be given by Eq. (19) and let ˆbn be the plug-in estimator from
Eq. (17). Then,
ˆbn = b + OP
 1/√n

.
(20)
Consequently the plug-in estimators ˆψi, ˆηi in Eq. (5) also have the same asymp-
totic error OP (1/√n).
The proof of Lemma 5 appears in the appendix.
Following it are some
remarks regarding the accuracy of various estimates as a function of the number
of classiﬁers and their accuracies. A detailed study of this issue is beyond the
scope of this paper.
4.2
A restricted-likelihood approach
The algorithm in Section 4.1 relied only on the ﬁrst three moments of the
data. We now present a second method to estimate the class imbalance, based
on a restricted likelihood function of all the data. This method is potentially
more accurate, however it requires the following stronger assumption of joint
conditional independence of all m classiﬁers,
Pr(f1 =a1, . . . , fm =am|y) =
m
Y
i=1
Pr(fi =ai|y).
(21)
It is important to note that under this assumption, the problem at hand
is equivalent to learning a mixture of two product distributions, addressed in
Freund and Mansour [1999]. For this problem, several recent works suggested
9

## Page 10

spectral tensor decomposition approaches, see Anandkumar et al. [2012a], Jain and Oh
[2013], Zhang et al. [2014].
In contrast, we now present a totally diﬀerent approach, not based on ten-
sor decompositions. Our starting point is Eq. (5) which provides consistent
estimates of ψ and η given the class imbalance b. In particular, any guess ˜b of
the class imbalance, yields corresponding guesses for the sensitivities and speci-
ﬁcities of all m classiﬁers, ˆψ(˜b) and ˆη(˜b). As described below, our approach is
to construct a suitable functional ˆGn(Z|˜b), that depends on both ˜b and on the
observed data Z, whose maxima as a function of ˜b, as n →∞is attained at the
true class imbalance b.
To this end, let f(x) = (f1(x), . . . , fm(x)) denote the vector of labels pre-
dicted by the m classiﬁers at an instance x. We deﬁne the following approximate
log-likelihood, assuming class imbalance ˜b
ˆgn(f(x)|˜b) = log Pr

f(x)| ˆ
ψ(˜b), ˆη(˜b),˜b

(22)
where ˆψ and ˆη are given by Eq. (5), and an expression for the above probability
is given in Eq. (38) in the appendix. Our functional ˆGn(Z|˜b) is the average of
ˆgn(f(x)|˜b) over all instances xj,
ˆGn(Z|˜b) = 1
n
n
X
j=1
ˆgn(f(xj)|˜b).
(23)
Note that the estimates of ψ, η in Eq. (5) become numerically unstable for b
close to ±1. Hence, in what follows we assume there is an a-priori known δ > 0,
such that the true class imbalance b ∈[−1 + δ, 1 −δ]. The estimate of the class
imbalance is then deﬁned as
ˆbn =
argmax
˜b∈[−1+δ,1−δ]
ˆGn(Z|˜b).
(24)
To justify Eq. (24), it is again constructive to consider the limit n →∞.
First, for any ˜b ∈[−1 + δ, 1 −δ], the convergence of ˆψ(˜b) and ˆη(˜b) to ψ(˜b) and
η(˜b), respectively, implies that at any instance x,
lim
n→∞ˆgn(f(x)|˜b) = g(f(x)|˜b) ≡log Pr(f(x)|ψ(˜b), η(˜b),˜b).
Next, since the n instances xj are i.i.d, by the law of large numbers, combined
with the delta method
lim
n→∞
ˆGn(Z|˜b) = G(˜b) ≡E(X,Y )
h
g(f(X)|˜b)
i
.
(25)
The following theorem, proven in the appendix, shows that the maxima of G(˜b)
is obtained at the true class imbalance ˜b = b, and that ˆbn →b in probability.
10

## Page 11

Algorithm 2 Estimating the class imbalance using the restricted likelihood
functional
1: Estimate the mean values {ˆµi}m
i=1, the covariance matrix ˆR, and the vector
ˆv.
2: for ˜b ∈(−1 + δ, 1 −δ) do
3:
Estimate ˆψ(˜b) and ˆη(˜b) via Eq. (5).
4:
Calculate ˆGn(Z|˜b) by Eqs. (22) and (23).
5: end for
6: Estimate b by Eq. (24).
Theorem 1. Assume all classiﬁer errors are independent, so Eq. (21) holds.
Let ǫ, δ > 0 be a-priori known, such that classiﬁers sensitivities and speciﬁcities
satisfy ǫ < ψi, ηi < 1−ǫ, and b ∈[−1 + δ, 1 −δ]. Then,
b =
argmax
˜b∈[−1+δ,1−δ]
E(X,Y )
h
g(f(X)|˜b)
i
(26)
and as n →∞the estimate ˆbn of Eq. (24) converges to b in probability.
Note that since ˆbn is the maximizer of a restricted likelihood, its convergence
to b is not a direct consequence of the consistency of ML estimators. Instead,
what is needed is uniform convergence in probability of ˆGn(˜b) to G(˜b), see
Newey [1991] and appendix. Also note that even though ˆGn(˜b) is not necessarily
concave, ﬁnding its global maxima requires optimization of a smooth function
of only one variable.
Algorithm 2 summarizes the method to estimate b by the restricted-likelihood
method. This algorithm scans possible values of ˜b, where each evaluation of ˆGn
requires O(mn) operations. Since ˆgn and consequently ˆGn are smooth functions
of ˜b in (−1 + δ, 1 −δ), the ﬁnite grid of values of ˜b can be of size polynomial in
n and the method is computationally eﬃcient.
5
The multi-class case
We now consider the multi-class case, with K > 2 classes. Here we are given
the predictions of m classiﬁers, fi : X →Y, where Y = {1, . . ., K}. Instead
of the class imbalance b, we now have a vector of K class probabilities pk =
Pr(Y = k). Similarly, instead of speciﬁcity and sensitivity, now each classiﬁer
is characterized by a K × K confusion matrix ψi
ψi
kk′ = Pr(fi(X) = k|Y = k′)
k, k′ ∈Y.
In analogy to Section 2, given only an m × n matrix of predictions, with
elements fi(xj) ∈{1 . . . K}, the problem is to estimate the confusion matrices
ψi of all classiﬁers and the class probabilities pk.
11

## Page 12

As in the binary case, we make an assumption regarding the mutual in-
dependence of errors made by diﬀerent classiﬁers. The precise independence
assumption (pairs, triplets or the full set of classiﬁers) depends on the method
employed.
By a simple reduction to the binary case, we now present a partial solution
to this problem. We develop a method to consistently estimate the class proba-
bilities pk and the diagonals of the confusion matrices, namely the probabilities
Pr(fi(X) = k|Y = k). However, we prove that even if the class probabilities are
a-priori known, estimating all entries of the m confusion matrices is not possible
via this binary reduction.
To this end, we build upon the methods developed in Sections 3 and 4 for
binary problems. Consider a split of the group Y = {1 . . .K} into two non-
empty disjoint subsets, Y = A ∪(Y \ A), where A ⊂Y is a non trivial subset of
Y, with 0 < |A| < K. Next, deﬁne the binary classiﬁers {f A
i }m
i=1:
f A
i (X) =

1
fi(X) ∈A
−1
fi(X) ̸∈A
Using one of the algorithms described in Section 4, we estimate the probability
of the group A
pA = Pr(Y ∈A) =
X
k∈A
pk
and the sensitivity of each classiﬁer f A
i
by Eq. (5).
In particular, when A = {k}, pA = pk and ψA
i = ψi
kk. Hence, by considering
all 1-vs.-all splits, we consistently and computationally eﬃciently estimate all
class probabilities pk, and all diagonal entries ψi
kk.
The following theorem, proven in the appendix, states a negative result, that
estimating the full confusion matrix is not possible by this binary reduction
method.
Theorem 2. Let µi
A = E[f A
i ] and let RA be the covariance matrix of the classi-
ﬁers {f A
i }m
i=1. The inverse problem of estimating the m confusions matrices ψi,
from the values of {µi
A}m
i=1 and RA for all possible subsets A of Y = {1 . . . K},
is in general ill posed with multiple solutions.
Theorem 2 implies that in order to completely estimate the confusion ma-
trices in a multiclass problem, it is necessary to use higher-order dependencies
such as tensors or even the full likelihood. Indeed, both Zhang et al. [2014] and
Jain and Oh [2013] derived such methods based on three-dimensional tensors.
While beyond the scope of this paper, we remark that combining our sim-
pler method with these tensor-based approaches might produce more accurate
algorithms for the multiclass case.
12

## Page 13

10
3
10
4
−0.2
0
0.2
0.4
0.6
0.8
n
(a) Estimating b via the 3-D tensor
T .
10
3
10
4
−0.2
0
0.2
0.4
0.6
0.8
n
(b) Estimating b via the restricted
likelihood ˆGn
Fig. 1: Mean and variance of the tensor-based and likelihood-based class imbal-
ance estimators vs. number of instances n, for several values of b.
6
Experiments
6.1
Artiﬁcial Data
First, we demonstrate the performance of the two class imbalance estimators on
artiﬁcial binary data. In the following we constructed an ensemble of m = 10
classiﬁers that make independent errors and thus satisfy Eq. (21). Their sen-
sitivities and speciﬁcities were chosen uniformly at random from the interval
[0.5, 0.8]. Thus, assumption (iii) on the balanced accuracies π holds. The vec-
tor of true labels y ∈{±1}n was randomly generated according to the class
imbalance b, and the data matrix Z was randomly generated according to y, ψ,
and η.
Fig. 1 presents the accuracy (mean and standard deviation) of the estimates
ˆb of the class imbalance, achieved by the two diﬀerent algorithms of Sections
4.1 and 4.2, vs. the number of unlabeled instances n, for several values of the
class imbalance, b = 0, 0.3, 0.6. As expected, the accuracy of both methods
improves with the number of instances. Fig. 2 shows the mean squared error
(MSE) E[(ˆb −b)2] vs. the number of samples n, on a log-log scale. The linear
line with slope ≈−1 shows that empirically ˆbn = b + OP (1/√n), in accordance
to Lemma 5. In addition, on simulated data, the restricted likelihood estimator
is more accurate than the tensor-based estimator.
6.2
Real data
We applied our algorithms on various binary and multi-class problems using a
total of 5 datasets: 4 datasets from the UCI repository [Bache and Lichman,
2013] and the MNIST data. Our ensemble consisted of m = 10 classiﬁcation
methods implemented in the software package Weka [Hall et al., 2009]. Due
to page limits, we present here results only on the ’magic’ dataset. Further
13

## Page 14

3
3.5
4
4.5
−3.8
−3.6
−3.4
−3.2
−3
−2.8
−2.6
−2.4
Mean Square error iter= 500
log10(n)
log10(MSE)
 
 
Tensor
Rest. Likelihood
Fig. 2: The MSE of the two class imbalance estimators vs. number of samples
on a log-log scale.
details on the diﬀerent datasets, classiﬁers and additional results appear in the
appendix.
The magic data contains 19, 000 instances with 11 attributes. The task is
to distinguish each instance as either background or high energy gamma rays.
Each of the m = 10 classiﬁers was trained on its own randomly chosen set
of 200 instances. The classiﬁers were then applied to the whole dataset, thus
providing the m × n prediction matrix. We compared the results of 4 diﬀerent
unsupervised ensemble methods: (i) Majority voting; (ii) SML of Parisi et al.
[2014]; (iii) i-SML as described in section 4; and (iv) Oracle ML: the MLE
formula (8) with the values of ψ and η, estimated from the full dataset with its
labels.
To assess the stability of the diﬀerent methods, for each dataset we repeated
the above simulation 30 times, each realization with diﬀerent randomly chosen
training sets. Fig. 3a shows the mean and standard deviation of the balanced
accuracy π achieved by the four methods on the ’magic’ dataset. It shows that
on average, i-SML improves upon the SML by approximately 2%, and both
are signiﬁcantly better than majority voting. Fig. 3b displays the error rates
1 −π i-SML vs.
1 −πSML for all 30 realizations.
As all points are below the
diagonal, the improvement over SML was consistent in all 30 simulation runs.
As shown in the appendix, similar results, and in particular the improvement
of i-SML over SML, were observed also in all 4 other datasets.
7
Summary and Discussion
In this paper we presented a simple spectral-based approach to estimate, in
an unsupervised manner, the accuracies of multiple classiﬁers, mainly in the
binary case. This, in turn, resulted in a novel unsupervised spectral ensemble
learner, denoted i-SML. The empirical results on several real data sets attest to
its competitive performance in practical situations where clearly the underlying
idealized assumptions that all classiﬁers make independent errors do not hold
exactly.
There are several interesting directions to extend this work. One possible
14

## Page 15

voting SML   i−SML oracle
0.7
0.72
0.74
0.76
0.78
0.8
0.82
(a) The balanced accuracies of
4 unsupervised ensemble methods
on the magic dataset.
0.2
0.22
0.24
0.26
0.2
0.21
0.22
0.23
0.24
0.25
0.26
1 −πSML
1 −πi−SML
(b) The empirical test error (1 −
πi-SML) vs. (1 −πSML) for 30 ran-
dom realizations.
Fig. 3: Comparing 4 unsupervised ensemble learning algorithms, based on m =
10 classiﬁers.
direction is to relax the strict assumptions of independence of classiﬁer errors
across all instances, for example by introducing the concept of instance diﬃculty.
A second interesting direction is the construction of novel semi-supervised en-
semble learners, when one is given not only the predictions of m classiﬁers on
a large unlabeled set of instances, but also their predictions on a small set of
labeled ones.
15

## Page 16

References
A. Anandkumar, R. Ge, D. Hsu, S.M. Kakade, and M. Telgarsky. Tensor decom-
positions for learning latent variable models. arXiv preprint arXiv:1210.7559,
2012a.
A. Anandkumar, D. Hsu, and S.M. Kakade. A method of moments for mixture
models and hidden markov models. arxiv preprint arxiv:1203.0683, 2012b.
K. Bache and M. Lichman. UCI machine learning repository, 2013.
A. P Dawid and A. M Skene. Maximum likelihood estimation of observer error-
rates using the em algorith. Journal of the Royal Statistical Society. Series
C, 28:20–28, 1979.
T.G. Dietterich. Ensemble methods in machine learning. In Lecture Notes in
Computer Science, volume 1857, pages 1–15. Springer, Berlin, 2000.
P. Donmez, G. Lebanon, and K. Balasubramanian. Unsupervised supervised
learning I: Estimating classiﬁcation and regression errors without labels. The
Journal of Machine Learning Research, 11:1323–1351, 2010.
J. Feldman, R. O’Donnell, and R.A. Servedio. Learning mixtures of product
distributions over discrete domains.
SIAM Journal on Computing, 37(5):
1536–1564, 2008.
Y. Freund and Y. Mansour. Estimating a mixture of two product distributions.
In COLT ’99 Proceedings of the twelfth annual conference on Computational
learning theory, pages 53–62, 1999.
M. Hall, E. Frank, G. Holmes, G. Pfahringer, P. Reutemann, and I.H Witten.
The weka data mining software: An update. SIGKDD Explorations, 11(1),
2009.
P. Jain and S. Oh. Learning mixtures of discrete product distributions using
spectral decompositions. arXiv preprint arXiv:1311.2972, 2013.
D.R. Karger, S. Oh, and D. Shah. Budget-optimal crowdsourcing using low-rank
matrix approximations.
In IEEE Alerton Conference on Communication,
Control and Computing, pages 284–291, 2011.
W. K. Newey. Uniform convergence in probability and stochastic equicontinuity.
Econometrica, 59:1161–1167, 1991.
F. Parisi, F. Strino, B. Nadler, and Y. Kluger. Ranking and combining multiple
predictors without labeled data.
Proceedings of the National Academy of
Sciences, 111:1253–1258, 2014.
E.A. Platanios, A. Blum, and T. Mitchell. Estimating accuracy from unlabeled
data. In Uncertainty in Artiﬁcial Intelligence, 2014.
16

## Page 17

V.C. Raykar, Y. Shipeng, L.H. Zhao, G.H. Valdez, C. Florin, L. Bogoni, and
Moy L. Learning from crowds. J. Machine Learning Research, 11:1297–1322,
2010.
L. Rokach. Collective-agreement-based pruning of ensembles. Computational
Statistics and Data Analysis, 53:1015–1026, 2009.
V.S. Sheng, F. Provost, and P.G. Ipeirotis. Get another label? improving data
quality and data mining using multiple, noisy labelers. In Proceedings of the
14th ACM SIGKDD international conference on Knowledge discovery and
data mining, pages 614–622, 2008.
A. Sheshadri and M. Lease. Square: A benchmark for research on computing
crowd concensus. In AAAI conference on human computation and crowd-
sourcing, 2013.
R. Snow, B. O’Connor, D. Jurafsky, and A.Y. Ng. Cheap and fast
but is it
good? In Conference on Empirical Methods in Natural Language Processing,
2008.
P. Welinder, S. Branson, S. Belongie, and P. Perona. The multidimensional
wisdom of crowds. In Advances in Neural Information Processing Systems 23
(NIPS 2010), 2010.
J. Whitehill, P. Ruvolo, T. Wu, J Bergsma, and J.R. Movellan. Whose vote
should count more: Optimal integration of labels from labelers of unknown
expertise. In Advances in Neural Information Processing Systems 22 (NIPS
2009), 2009.
Y. Zhang, X. Chen, D. Zhou, and M.I. Jordan. Spectral methods meet em: A
provably optimal algorithm for crowdsourcin. arXiv preprint arXiv:1406.3824,
2014.
17

## Page 18

A
Estimation of ψ and η
Proof of Lemma 2. We ﬁrst recall the following formula, derived in Parisi et al.
[2014], for the vector µ containing the mean values of the m classiﬁers,
µ = 2δ + b(2π −1)
(27)
where δ = (δ1, . . . , δm) denotes the vector containing half the diﬀerence between
ψ and η,
δ = ψ −η
2
.
(28)
Next, recall from Lemma 1 (also proven in Parisi et al. [2014]) that the oﬀ-
diagonal elements of the covariance matrix R correspond to a rank-1 matrix
vvT where,
v =
p
1 −b2(2π −1).
(29)
Inverting the relation between v and π in Eq. (29) gives
π = 1
2

v
√
1 −b2 + 1

.
(30)
Plugging (30) into (27), we obtain the following expression for the vector δ, in
terms of v and µ,
δ = 1
2

µ −b
v
√
1 −b2

.
(31)
Combining (28), (30) and (31) we obtain ψ(b) and η(b),
ψ = π + δ = 1
2
 
1 + µ + v
r
1 −b
1 + b
!
,
η = π −δ = 1
2
 
1 −µ + v
r
1 + b
1 −b
!
.
B
Statistical Properties of ψ and η
Proof of Lemma 3. Eq. (5) provides an explicit expression for ˆψ and ˆη as a
function of the estimates ˆv and ˆµ. The empirical mean ˆµ is clearly not only
unbiased, but by the law of large numbers also a consistent estimate of µ, and
its error indeed satisﬁes
ˆµ = µ + OP
 1
√n

.
The estimate ˆv, computed by one of the methods described in Parisi et al. [2014]
may be biased, but as proven there is still consistent, and assuming at least three
18

## Page 19

classiﬁers are diﬀerent than random (in particular, implying that the eigenvalue
of the rank one matrix is non-zero), its error also decreases as OP

1
√n

,
ˆv = v + OP
 1
√n

.
Given the exact value of the class imbalance b, since the dependency of ˆψ and
ˆη on ˆv and ˆµ is linear, it follows that both are also consistent and that their
estimation error is OP

1
√n

.
C
The joint covariance tensor T
Proof of Lemma 4. To simplify the proof, we ﬁrst introduce the following linear
transformation to the original classiﬁers,
˜fi(x) = fi(x) + 1
2
.
Note, that the output space Y of the new classiﬁers is {0, 1}, with class prob-
abilities equal to 1 −p and p respectively. Let us also denote by ˜ηi and ˜ψi the
following probabilities,
˜ηi = Pr( ˜fi(x) = 1|Y = 0), ˜ψi = Pr( ˜fi(x) = 1|Y = 1).
Note that ˜ηi is not the speciﬁcity of classiﬁer i, but rather its complement,
˜ηi = 1 −ηi.
The mean of classiﬁer ˜fi, denoted ˜µi, is given by
˜µi = E[ ˜fi(X))] = Pr( ˜fi(X) = 1) = p ˜ψi + (1 −p)˜ηi
(32)
Next, let us calculate the (un-centered) covariance between two diﬀerent classi-
ﬁers i ̸= j,
E[ ˜fi(X) ˜fj(X)] = Pr( ˜fi(X) = 1, ˜fj(X) = 1)
= p ˜ψi ˜ψj + (1 −p)˜ηi˜ηj
(33)
Last, the joint covariance between 3 diﬀerent classiﬁers i ̸= j ̸= k is given by
E[ ˜fi(X) ˜fj(X) ˜fk(X)] = Pr( ˜fi(X)= ˜fj(X)= ˜fk(X)=1)
= p ˜ψi ˜ψj ˜ψk + (1 −p)˜ηi˜ηj ˜ηk
(34)
The ﬁrst step in calculating the joint covariance tensor of the original clas-
siﬁers is to note that fi = 2 ˜fi −1 and µi = 2˜µi −1. Hence,
Tijk = E[(fi(X) −µi)(fj(X) −µj)(fk(X) −µk)] = 8 ˜Tijk
where
˜Tijk = E[( ˜fi(X) −˜µi)( ˜fj(X) −˜µj)( ˜fk(X) −˜µk)].
19

## Page 20

Upon opening the brackets, the latter can be equivalently written as
˜Tijk = E
h
˜fi(X) ˜fj(X) ˜fk(X)
i
−˜µiE
h
˜fj(X) ˜fk(X)
i
−˜µjE
h
˜fi(X) ˜fk(X)
i
−˜µkE
h
˜fi(X) ˜fj(X)
i
+ 2˜µi˜µj ˜µk
(35)
Plugging (32),(33) and (34) into (35) we get,
˜Tijk = p ˜ψi ˜ψj ˜ψk + (1 −p)˜ηi˜ηk˜ηj−

p ˜ψi + (1 −p)˜ηi
 
p ˜ψj ˜ψk + (1 −p)˜ηj ˜ηk

−

p ˜ψj + (1 −p)˜ηj

p ˜ψk ˜ψi + (1 −p)˜ηk˜ηi

−

p ˜ψk + (1 −p)˜ηk

p ˜ψi ˜ψj + (1 −p)˜ηi˜ηj

+
2

p ˜ψi + (1 −p)˜ηi
 
p ˜ψj + (1 −p)˜ηj
 
p ˜ψk + (1 −p)˜ηk

Opening the brackets and collecting similar terms yields
˜Tijk = (p −3p2 + 2p3) ˜ψi ˜ψj ˜ψk+
 2p2(1 −p) −p(1 −p)
 
˜ηi ˜ψj ˜ψk + ˜ηj ˜ψk ˜ψi + ˜ηk ˜ψi ˜ψj

+
 2p(1 −p)2 −p(1 −p)
 
˜ηi˜ηj ˜ψk + ˜ηj ˜ηk ˜ψi + ˜ηk˜ηi ˜ψj

+
 (1 −p) −3(1 −p)2 + 2(1 −p)3
˜ηi˜ηk˜ηj.
Note that all polynomials in p in the above expression are equal to ±p(1−p)(1−
2p). Hence,
˜Tijk =p(1 −p)(1 −2p)( ˜ψi ˜ψj ˜ψk −˜ηi ˜ψj ˜ψk −˜ηj ˜ψk ˜ψi−
˜ηk ˜ψi ˜ψj + ˜ηi˜ηj ˜ψk + ˜ηj ˜ηk ˜ψi + ˜ηk˜ηi ˜ψj −˜ηi˜ηk˜ηj)
(36)
Finally, replacing ˜ψi = ψi, ˜ηi = 1 −ηi and p = 1+b
2 , yields
Tijk = −2b(1 −b2)(ψi + ηi −1)(ψj + ηj −1)(ψk + ηk −1)
= −2b(1 −b2)(2πi −1)(2πj −1)(2πk −1).
Proof of Lemma 5. To prove that ˆbn is consistent with an asymptotic error
OP (1/√n), we ﬁrst recall that according to Parisi et al. [2014], it follows that
ˆv = v + OP
 1
√n

.
20

## Page 21

By its deﬁnition, each entry of ˆTijk also incurs an error of OP (1/√n). Hence, by
the delta method, the estimate ˆα of Eq. (19), being a least squares minimizer,
also satisﬁes
ˆα = α + OP (1/√n).
Since ˆbn is found by the smooth relation of Eq. (17), again by the delta method,
ˆbn = b+OP (1/√n). Finally, the fact that the corresponding estimates ˆψi and ˆηi
also have errors OP (1/√n) follows by standard application of the delta method
to Eq. (5), where all quantities ˆµ, ˆv and ˆb have errors OP (1/√n).
Dependence of estimated parameters on number of classiﬁers and
their accuracies.
Beyond the fact that ˆα and consequently ˆbn, ˆψ, ˆη are all
O(1/√n) consistent, it is of interest to study the dependence of these estimates
on the number of classiﬁers and their accuracies. To this end, we ﬁrst prove the
following simple result.
Lemma 6. Let ˆα be the estimate of α in Eq. (19). Then asymptotically as
n →∞, its estimation error is given by
ˆα −α = ⟨ˆT −T, v⊗3⟩
⟨v⊗3, v⊗3⟩
−α⟨ˆv⊗3 −v⊗3, v⊗3⟩
⟨v⊗3, v⊗3⟩
+ OP
 1
n

(37)
where v⊗3 = v⊗v⊗v, and for any two tensors T, S, ⟨T, S⟩= P
i<j<k TijkSijk.
Proof. The minimizer of Eq. (19) is given by
ˆα =
⟨ˆT, ˆv⊗3⟩
⟨ˆv⊗3, ˆv⊗3⟩
According to Parisi et al. [2014], as n →∞, the estimate ˆv is O(1/√n) consis-
tent, namely ˆv = v + δv, where δv = OP (1/√n). Writing ˆT = T + ( ˆT −T )
where the latter is also OP (1/√n) and inserting these into the expression for ˆα
above gives that
ˆα = ⟨T, v⊗3⟩+ ⟨ˆT −T, v⊗3⟩+ ⟨T, ˆv⊗3 −v⊗3⟩+ OP (1/n)
⟨v⊗3, v⊗3⟩+ 2⟨v⊗3, ˆv⊗3 −v⊗3⟩+ OP (1/n)
Next, recall that T = αv⊗3. Now, keeping only the leading order error terms
yields Eq. (37).
According to Eq. (37), the estimation error depends on the statistical prop-
erties of the deviations ˆv −v and ˆT −T and their correlations. While these
are quite complicated, we may gain insight by looking at some particular in-
stances. Assume for simplicity that all classiﬁers have comparable accuracies.
Then, ⟨v⊗3, v⊗3⟩∝m(m −1)(m −2)/6 · (2π −1)6. Hence, the estimation error
in ˆα should decrease with the number of classiﬁers. Moreover, for a balanced
problem with b = 0 and hence α = 0, to leading order, the errors in ˆα and conse-
quently also in ˆbn should not depend on the errors in estimating the eigenvector
21

## Page 22

0.9
1
1.1
1.2
1.3
−2.1
−2.05
−2
−1.95
−1.9
−1.85
−1.8
Mean Absolute error E[|ˆb −b|]
log10(m)
log10(MAE)
 
 
b=0, Tensor
b=0, Oracle
b=0.3, Tensor
b=0.3, Oracle
Fig. 4: Mean absolute error for the tensor based method, E[|ˆbn −b|] vs. number
of classiﬁers m, on log-log scale.
v. Figure 4 shows this empirically. The x-axis is the number of classiﬁers, the
y-axis is the mean absolute deviation E[|ˆbn −b|] (MAE), both on a log scale. We
considered two values b = 0 and b = 0.3, and for each value of b we plotted two
curves, one corresponding to the estimate ˆb computed from ˆα based on ˆv, and
the second, an “oracle” one, where ˆα is estimated using the true v. Indeed, for
b = 0 both curves nearly coincide, in accordance to Eq. (37). In this simulation,
all classiﬁers had a balanced accuracy in the range [0.69, 0.71], and n = 10, 000.
These results suggest that it is potentially proﬁtable to estimate the eigenvector
v and the scalar α jointly from both the covariance matrix ˆR and the tensor ˆT,
and not separately as done in the present paper. This, as well as a more detailed
study of the estimation errors are issues beyond the scope of the current work.
D
The Restricted Likelihood Function
Proof of Theorem 1. By deﬁnition, the function ˆgn(f(x)|˜b) in Eq. (22) is the
log-likelihood of the observed vector f(x) of predicted labels at an instance x,
assuming the class imbalance is ˜b and using the estimates ˆψ and ˆη for the
sensitivities and speciﬁcities of the m classiﬁers.
Under the assumption that all classiﬁers make independent errors, the ex-
pression for Pr(f(x)| ˆ
ψ, ˆη,˜b) is given by
Pr(f|˜b) = Pr(y = 1|˜b) Pr(f|˜b, y = 1)+
Pr(y = −1|˜b) Pr(f|˜b, y = −1) =

1+˜b
2
 m
Y
i=1
ˆψ
1+fi(x)
2
i
(1 −ˆψi)
1−fi(x)
2
+

1−˜b
2
 m
Y
i=1
ˆη
1−fi(x)
2
i
(1 −ˆηi)
1+fi(x)
2
(38)
22

## Page 23

We ﬁrst prove Eq. (26), that upon using the exact log-likelihood function g(f|˜b),
its mean is maximized at the true value b. To this end, we write the expectation
explicitly,
E[g(f|˜b)]
=
X
f∈{−1,1}m
Pr(f|b)g(f|˜b)
=
X
f∈{−1,1}m
Pr(f|b) log Pr(f|˜b)
(39)
Note the diﬀerence between the assumed class imbalance ˜b, which appears inside
the logarithm, and its true value b, over which we take the expectation.
To prove Eq. (26), let us ﬁrst present the following auxiliary lemma, which
can be easily proved using Lagrange multipliers.
Lemma 7. Consider the following function of k unknown variables {ci}k
i=1,
h({ci}k
i=1|{ai}k
i=1) =
k
X
i=1
ai log(ci).
(40)
where {ai}k
i=1 are k non-negative constants. Under the constraints that Pk
i=1 ci =
1, and ci ≥0, the function h has a global maxima at ci = ai for all i.
We use this lemma with k = 2m and the following set of 2m constants
af(b) = Pr(f|b), over all possible m-dimensional vectors f ∈{−1, 1}m, and the
2m variables cf = Pr(f|˜b). The expectation of g is now equal to
G(˜b) = E[g(f|˜b)] =
2m
X
i=1
ai log(ci)
(41)
By Eq. (40), over all possible choices of ci, the expectation attains its maxima at
ci = ai for all i. Since at ˜b = b, the corresponding probabilities Pr(f|˜b = b) = af,
Eq. (26) follows.
Next, we wish to prove that ˆbn →b in probability. To this end, we follow the
approach outlined in Newey [1991], and prove the following uniform convergence
in probability of ˆGn to G,
sup
˜b∈[−1+δ,1−δ]
| ˆGn(˜b) −G(˜b)| = oP (1)
This equation, coupled with the equicontinuity of G implies the convergence in
probability of the maximizer of ˆGn (namely ˆbn) to that of G, which by Eq. (26)
is b.
As proved in [Newey, 1991, Theorem 2.1], this uniform convergence in prob-
ability is satisﬁed if and only if there is pointwise convergence of ˆGn(˜b) to G(˜b),
and ˆGn(˜b) is stochastic equicontinuous. Fortunately, a suﬃcient condition for
the latter property is that ˆGn(˜b) is continuously diﬀerentiable and its derivative
bounded, see Newey [1991] Corollary 2.2 and discussion after it.
23

## Page 24

In our case, since ˆGn(˜b) = 1/n P
i ˆgn(f(xi)|˜b), it suﬃces to prove that for
any vector f, the function ˆgn(f|˜b) is continuously diﬀerentiable with a bounded
derivative.
First note that by their deﬁnition, Eq.
(5), the functions ˆψi(˜b)
and ˆηi(˜b) are continuously diﬀerentiable with bounded derivative for all ˜b ∈
[−1 + δ, 1 −δ]. Next, under the assumptions of the theorem, that ψi and ηi are
ǫ bounded from 0 and from 1, and hence also their estimates can be restricted
to ǫ < ˆψi, ˆηi < 1 −ǫ, the term inside the logarithm in Eq. (22) is bounded away
from zero. Hence, by its deﬁnition ˆgn satisﬁes the required condition.
E
Ambiguity in the Multi-Class Case
Proof of Theorem 2. For simplicity, let us assume that all K class probabilities
are equal, pi =
1
K for i = 1, . . . , K. Let fi be the set of original classiﬁers with
confusion matrices {ψi}m
i=1. We shall now construct another set of classiﬁers
with diﬀerent confusion matrices that nonetheless lead to the same values µi
A
and RA for all subsets A.
To this end, assume that all entries of the ﬁrst confusion matrix ψ1 are
strictly positive and strictly smaller than one. Consider a second set of confusion
matrices { ˜ψi}m
i=1 identical to the ﬁrst, except for the following six changes in
ψ1: For three ﬁxed indices j ̸= k ̸= l, let
˜ψ1
jk = ψ1
jk + ∆
˜ψ1
kj = ψ1
kj −∆
˜ψ1
lj = ψ1
lj + ∆
˜ψ1
jl = ψ1
jl −∆
˜ψ1
kl = ψ1
kl + ∆
˜ψ1
lk = ψ1
lk −∆
where ∆is suﬃciently small so that all entries of ˜ψ1 are in [0, 1].
Note that the new matrix ˜ψ1 is a valid confusion matrix, since for any column
r ∈{1, . . . , K}
K
X
i=1
˜ψ1
ir = 1.
Let ˜f1 be the classiﬁer corresponding to the modiﬁed matrix ˜ψ1. Next, note that
the ﬁrst order statistics of ˜f1 and of f1 are unchanged. Indeed, by deﬁnition
Pr( ˜f1(X) = r) = 1
K
K
X
i=1
˜ψ1
ri
If r /∈{j, k, l}, then ˜ψ1
ri = ψ1
ri and thus
Pr( ˜f1(X) = r) = Pr(f1(X) = r)
(42)
If r ∈{j, k, l}, then by construction, in the r-th row of ˜ψ1 there are precisely
two modiﬁed entries, one increased by ∆and the other reduced by ∆, so overall
the above equation still holds. Eq. (42) directly implies that ˜µ1
A = µ1
A for all
subsets A.
24

## Page 25

Next, let us show that the covariance matrices RA also remain unchanged.
Recall that the entries of RA are determined by the values ψ1
A . . . ψm
A and
η1
A . . . ηm
A . Hence, it suﬃces to show that for all subsets A
˜ψ1
A = ψ1
A
and
˜η1
A = η1
A
(43)
To this end, recall that by deﬁnition
˜ψ1
A = 1
K
X
i,i′∈A
˜ψ1
ii′
and
˜η1
A = 1
K
X
i,i′ /∈A
˜ψ1
ii′
First consider the case |A ∩{j, k, l}| = 0. Here, all relevant entries in the sum
for ˜ψ1
A are unchanged. In contrast, the sum for ˜η1
A includes all six modiﬁed
entries. Both sums remain unchanged, and so Eq. (43) holds.
The proof for the other cases, where A ∩{j, k, l} ̸= ∅follows similar argu-
ments.
To conclude, both {ψi}m
i=1 and { ˜ψi}m
i=1 have the same values µi
A and covari-
ance matrices RA.
F
Ensemble of Machine Learning Classiﬁers
Table 1 presents the 10 diﬀerent classiﬁers used in our experiments. For each
dataset, each classiﬁer was trained with 200 diﬀerent (randomly chosen) in-
stances.
G
Real Datasets
We tested our methods on a total of ﬁve datasets, 4 from the UCI repository
and the MNIST digits data. A short description of each of the datasets is given
in Table 2. A comparison of the performance of various ensemble learners on
these datasets appears in Fig. 5.
25

## Page 26

classiﬁer
Weka library
IBk - K nearest
neighbours, K = 1
lazy.IBk
KStar - Instance
based classiﬁer
lazy.KStar
J48 - Decision tree
trees.J48
PART - Partial decision
trees classiﬁer
rules.PART
LMT - Logistic model
trees
trees.LMT
Random forest -
with n = 10 trees
trees.RandomForest
Logistic Regression
functions.SimpleLogistic
Decision Stump -
One level decision tree
trees.DecisionStump
Sequential Minimal
Optimization
functions.SMO
NaiveBayes
bayes.NaiveBayes
Table 1: 10 classiﬁcation methods implemented in the software package Weka.
dataset Task
instances
attributes
Magic
classifying
gamma
rays
from
back-
ground noise
19000
11
Spam
classifying
spam
from regular mail
4600
57
Musk
classifying diﬀerent
types of molecules
to be ’musk’ or ’non
musk’
6600
88
Miniboo distinguish electron
neutrinos
(signal)
from muon neutri-
nos (background)’
130000
50
Mnist
To deﬁne a binary
problem,
we
di-
vided the MNIST
data set into two
classes as follows:
0 −4 vs. 5 −9
40000
282
Table 2: Properties of datasets from the UCI repository
26

## Page 27

voting SML   i−SML oracle
0.89
0.9
0.91
0.92
0.93
0.94
(a) Spam Dataset
voting SML   i−SML oracle
0.75
0.8
0.85
0.9
0.95
(b) Musk Database
voting
SML   i_SML 
oracle
0.75
0.8
0.85
0.9
(c) Miniboo Dataset
sml   
i−sml 
oracle
0.83
0.835
0.84
0.845
0.85
(d) ’MNIST’ Dataset.
Fig. 5: The balanced accuracies of 4 unsupervised ensemble learning algorithms,
all with m = 10 classiﬁers. In panel 5d we do not show the accuracy of majority
voting which was signiﬁcantly lower than all others.
27
