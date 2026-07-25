---
source_pdf: Tenzer2022_Crowdsourcing_Regression_Spectral.pdf
slug: tenzer2022-crowdsourcing-regression-spectral
pages: 14
extracted_on: 2026-07-13
---

# Tenzer2022_Crowdsourcing_Regression_Spectral

## Page 1

arXiv:1703.02965v1  [stat.ML]  8 Mar 2017
Unsupervised Ensemble Regression
Omer Dror
OMER.DROR@WEIZMANN.AC.IL
Weizmann Institute of Science
Boaz Nadler
BOAZ.NADLER@WEIZMANN.AC.IL
Weizmann Institute of Science
Erhan Bilal
EBILAL@US.IBM.COM
IBM Thomas J. Watson Research Center
Yuval Kluger
YUVAL.KLUGER@YALE.EDU
Yale University School of Medicine
Abstract
Consider a regression problem where there is no labeled
data and the only observations are the predictions fi(xj) of
m experts fi over many samples xj. With no knowledge on
the accuracy of the experts, is it still possible to accurately
estimate the unknown responses yj? Can one still detect
the least or most accurate experts? In this work we propose
a framework to study these questions, based on the assump-
tion that the m experts have uncorrelated deviations from
the optimal predictor. Assuming the ﬁrst two moments of
the response are known, we develop methods to detect the
best and worst regressors, and derive U-PCR, a novel prin-
cipal components approach for unsupervised ensemble re-
gression. We provide theoretical support for U-PCR and il-
lustrate its improved accuracy over the ensemble mean and
median on a variety of regression problems.
1. Introduction
Consider the following unsupervised ensemble regression
setup: The only observations are an m × n matrix of real-
valued predictions fi(xj) made by m different regressors
or experts {fi}m
i=1, on a set of unlabeled samples {xj}n
j=1.
There is no a-priori knowledge on the accuracy of the ex-
perts and no labeled data to estimate it. Given only the
above observed data and minimal knowledge about the un-
observed response, such as its mean and variance, is it pos-
sible to (i) rank the m regressors, say by their mean squared
error; or at least detect the most and least accurate ones?
and (ii) construct an ensemble predictor for the unobserved
continuous responses yj, more accurate than both the in-
dividual predictors and simple ensemble strategies such as
their mean or median?
Our motivation for studying this problem comes from sev-
eral application domains, where such scenarios naturally
arise. Two such domains are biology and medicine, where
in recent years there are extensive collaborative efforts to
solve challenging prediction problems, see for example the
past and ongoing DREAM competitions1.
Here, multi-
ple participants construct prediction models based on pub-
lished labeled data, which are then evaluated on held-out
data whose statistical distribution may differ signiﬁcantly
from the training one.
A key question is whether one
can provide more accurate answers than those of the in-
dividual participants, by cleverly combining their predic-
tion models. In the experiment section 5 we present one
such example, where competitors had to predict the con-
centrations of multiple phosphoproteins in various cancer
cell lines (Hill et al., 2016a). Understanding the causal re-
lationships between these proteins is important as it may
explain variation in disease phenotypes or therapeutic re-
sponse (Hill et al., 2016b). A second application comes
from regression problems in computer vision. A speciﬁc
example, also described in Section 5, is accurate estimation
of the bounding box around detected objects in images by
combining several pre-constructed deep neural networks.
The regression problem we consider in this paper is a par-
ticular instance of unsupervised ensemble learning. Mo-
tivated in part by crowdsourced labeling tasks, previ-
ous works on unsupervised ensemble learning mostly fo-
cused on discrete outputs, considering binary, multiclass
or ordinal classiﬁcation (Johnson, 1996; Sheng et al., 2008;
Whitehill et al., 2009; Raykar et al., 2010; Platanios et al.,
2014; 2016; Zhou et al., 2012). Dawid and Skene (1979)
were among the ﬁrst to consider the problem of unsuper-
vised ensemble classiﬁcation. Their approach was based on
the assumption that experts make independent errors con-
1www.dreamchallenges.org

## Page 2

Unsupervised Ensemble Regression
ditioned on the unobserved true class label. Even for this
simple model, estimating the experts’ accuracies and the
unknown labels via maximum likelihood is a non-convex
problem, typically solved by the expectation-maximization
algorithm. Recently, several authors proposed spectral and
tensor based methods that are computationally efﬁcient
and asymptotically consistent (Anandkumar et al., 2014;
Zhang et al., 2014; Jaffe et al., 2015).
In contrast to the discrete case, nearly all previous works
on ensemble regression considered only the supervised set-
ting. Some ensemble methods, such as boosting and ran-
dom forest are widely used in practice.
In this work we propose a framework to study unsupervised
ensemble regression, focusing on linear aggregation meth-
ods. In Section 2, we ﬁrst review the optimal weights that
minimize the mean squared error (MSE) and highlight the
key quantities that need to be estimated in an unsupervised
setting. Next, in Section 3 we describe related prior work
in supervised and unsupervised regression.
Our main contributions appear in Section 4. We propose a
framework for unsupervised ensemble regression, based on
an analogue of the Dawid and Skene classiﬁcation model,
adapted to the regression setting. Speciﬁcally, we assume
that the m experts make approximately uncorrelated er-
rors with respect to the optimal predictor that minimizes
the MSE. We show that if we knew the minimal attainable
MSE, then under our assumed model, the accuracies of the
experts can be consistently estimated by solving a system
of linear equations. Next, based on our theoretical anal-
ysis, we develop methods to estimate this minimal MSE,
detect the best and worst regressors and derive U-PCR, a
novel unsupervised principal components ensemble regres-
sor. Section 5 illustrates our methods and the improved ac-
curacy of U-PCR over the ensemble mean and median, on
a variety of regression problems. These include both prob-
lems for which we trained multiple regression algorithms,
as well as the two applications mentioned above where the
regressors were constructed by a third party and only their
predictions were given to us.
Our main ﬁndings are that given only the predictions fi(xj)
and the ﬁrst two moments of the response: (i) our approach
is able to distinguish between hard prediction problems
where any linear aggregation of the m regressors yields
large errors, and feasible problems where a suitable linear
combination of the regressors can accurately estimate the
response; (ii) our ranking method is able to reliably detect
the most and least accurate experts; and (iii) quite consis-
tently, U-PCR performs as well as and sometimes signif-
icantly better than the mean and median of the m regres-
sors. We conclude in Section 6 with a summary and future
research directions in unsupervised ensemble regression.
2. Problem Setup
Consider a regression problem with a continuous response
Y ∈R and explanatory features X from an instance space
X.
Let {f1, . . . , fm} be m pre-constructed regression
functions, fi : X →R, interchangeably also called ex-
perts, and let {xj}n
j=1 be n i.i.d. samples from the marginal
distribution of X. We consider the following unsupervised
ensemble regression setting, in which the only observed
data is the m × n matrix of predictions



f1(x1)
· · ·
f1(xn)
...
...
...
fm(x1)
· · ·
fm(xn)


.
(1)
In particular, there are no labeled data pairs (xj, yj) and no
a-priori knowledge on the accuracy of the m regressors.
Given only the matrix (1), and explicit knowledge of the
ﬁrst two moments of Y , we ask whether it is possible to: (i)
estimate the accuracies of the m experts, or at least identify
the best and worst of them, and (ii) accurately estimate the
responses yj by an ensemble method ˆy : {fi(x)}m
i=1 7→R,
whose input are the predictions of f1, . . . , fm. As we ex-
plain below knowing the ﬁrst two moments of Y seems
necessary as otherwise the data matrix (1) can be arbitrar-
ily shifted and scaled. Such knowledge is reasonable in
various settings, for example from past experience, previ-
ous observations or physical principles.
Following the literature on supervised ensemble regression,
we consider linear ensemble learners. Speciﬁcally, we re-
strict ourselves to the following subclass
ˆyw(x) = θ1 +
m
X
i=1
wi
 fi(x) −µi

(2)
where θ1 = E[Y ] and µi = E[fi(X)] are assumed known,
and w = (w1, . . . , wm)T . Note that in this subclass, for
any vector w, E(X,Y )

ˆyw(X)

= θ1. While µi is typically
unknown, it can be accurately estimated given the predic-
tions of fi in Eq. (1) and provided n ≫1.
As our risk measure, we use the popular mean squared error
MSE = E[(Y −ˆy(X))2]. For completeness, we ﬁrst review
the optimal weights under this risk and describe several su-
pervised ensemble methods that estimate them.
Optimal Weights.
Let C be the m×m covariance matrix
of the m regressors with elements
Cij = E[(fi(X) −µi)(fj(X) −µj)] ,
(3)
and let ρ = (ρ1, . . . , ρm)T be the vector of covariances
between the individual regressors and the true response,
ρi = E(X,Y )[(Y −θ1)(fi(X) −µi)] .
(4)

## Page 3

Unsupervised Ensemble Regression
Let w∗be a weight vector that minimizes the MSE
w∗= argmin
w
E(X,Y )
h ˆyw(X) −Y
2i
(5)
Then it is easy to show that:
Lemma 1. The weights w∗satisfy
ρ = Cw∗.
(6)
Note that w∗depends only on ρ and C. If the m ensemble
regressors are linearly independent, then C is invertible and
w∗is unique. In our unsupervised scenario, the matrix C
can be estimated from the predictions fi(xj). In contrast,
estimating ρ directly from its deﬁnition in Eq. (4) requires
labeled data. A key challenge in unsupervised ensemble
regression is thus to estimate ρ without any labeled data.
3. Previous Work
This section provides a brief overview of prior art, ﬁrst
methods for unsupervised ensemble regression, and then
two supervised ensemble regression methods that are re-
lated to our approach.
We conclude this section with
the popular Dawid-Skene model of unsupervised ensemble
classiﬁcation, also relevant to our work.
3.1. Unsupervised Ensemble Regression
Whereas many works considered unsupervised ensem-
ble classiﬁcation, far fewer studied the regression case.
Donmez et al. (2010), proposed a general framework called
unsupervised-supervised learning. In the case of regres-
sion, they assumed that the marginal probability density
function of the response p(y) is known and that the regres-
sors follow a known parametric model with parameter θ. In
this setup, given only unlabeled data, θ can be estimated
by maximum likelihood. In contrast, our approach is far
more general as we do not assume a parametric model, nor
knowledge of the full marginal density p(y).
More closely related is the recent work of Wu et al. (2016),
which in turn is based on Ionita-Laza et al. (2016) and
(Parisi et al., 2014). Here, the authors compute the lead-
ing eigenvector of the covariance of the m regressors, and
use it both to detect inaccurate regressors and to determine
the weights of the accurate ones. However, as Wu et al.
(2016) themselves write, this relation between the leading
eigenvector and regressor accuracy “is based on intuition,
and we do not have a rigorous mathematical proof so far”.
Our work provides a solid theoretical support for a variant
of this spectral approach.
3.2. Supervised Ensemble Regression
As reviewed by Mendes-Moreira et al. (2012), quite a few
supervised ensemble regressors were proposed over the
past 30 years.
These can be broadly divided into two
groups.
Methods in the ﬁrst group re-train a basic re-
gression algorithm multiple times on different subsets of
the labeled data, possibly also assigning weights to the
various labeled instances.
Examples include stacking
(Wolpert, 1992; Breiman, 1996; Leblanc and Tibshirani,
1996),
random forest (Breiman, 2001) and boosting
(Freund and Schapire, 1995; Friedman et al., 2000).
In contrast, ensemble methods in the second group view the
regressors as pre-constructed and only estimate the weights
of their linear combination.
Perrone and Cooper (1992)
and Merz and Pazzani (1999) derived two such methods,
which we brieﬂy describe below.
While not directly related, there is also extensive litera-
ture on supervised combination of forecasts in time series
analysis and on methods to combine multiple estimators,
see Timmermann (2006); Lavancier and Rochet (2016) and
many references therein.
3.3. Generalized Ensemble Method
Perrone and Cooper (1992) were among the ﬁrst to con-
sider supervised ensemble regression.
They deﬁned the
misﬁt of predictor i as mi(x) = fi(x)−y, and proposed the
Generalized Ensemble Method (GEM), with P
i wi = 1,
ˆyGEM(x) =
X
i
wifi(x) = y +
X
i
wimi(x).
The corresponding weights that minimize the MSE are
wGEM
i
=
X
j
C∗−1
ij
. X
j,k
C∗−1
jk .
(7)
where C∗is the m×m misﬁt population covariance matrix
C∗
ij = E(X,Y )[mi(X)mj(X)] .
(8)
Perrone and Cooper (1992) proposed to estimate the un-
known matrix C∗and consequently wGEM using a labeled
set {(xi, yi)}ntrain
i=1 . Unfortunately, in many practical scenar-
ios multi-colinearity between the m regressors leads to an
ill conditioned matrix C∗, that cannot be robustly inverted.
3.4. PCR*
A common approach to handle ill conditioned multivariate
problems is via principal component regression (Jolliffe,
2002).
In the context of supervised ensemble learning,
Merz and Pazzani (1999) suggested such a method, de-
noted PCR*. Given a labeled set {(xi, yi)}ntrain
i=1 let ˆC be
the m × m sample covariance matrix of the m regressors,
ˆCij =
1
ntrain
ntrain
X
k=1
 fi(xk) −ˆµi
 fj(xk) −ˆµj


## Page 4

Unsupervised Ensemble Regression
where ˆµi =
1
ntrain
Pntrain
j=1 fi(xj), and let v1, . . . , vK be the
top K leading eigenvectors of ˆC. Merz and Pazzani (1999)
proposed a weight vector of the form w = PK
k=1 akvk,
with coefﬁcients ak determined by least squares regression
over the training set. The number of principal components
K is chosen by minimizing V -fold cross validation error.
In the common scenario where some ensemble regressors
are highly correlated, the matrix C∗is ill-conditioned. The
GEM estimator, which inverts ˆC∗then yields unstable
predictions.
In contrast, PCR* with a small number of
components can be viewed as a regularized method, pro-
viding stability and robustness.
In a supervised setting,
Merz and Pazzani (1999) found PCR* to outperform GEM.
3.5. Unsupervised Ensemble Classiﬁcation
The simplest model for unsupervised ensemble classiﬁca-
tion, going back to Dawid and Skene (1979) is that condi-
tional on the label Y , classiﬁers make independent errors
Pr
 fi(X), fj(X)|Y

= Pr(fi(X)|Y ) · Pr(fj(X)|Y ).
(9)
Dawid and Skene (1979) estimated the classiﬁer accu-
racies and the labels by the EM method.
In recent
years several authors developed computationally efﬁcient
and rate optimal methods to estimate these quantities
(Anandkumar et al., 2014; Zhang et al., 2014; Jaffe et al.,
2015).
To the best of our knowledge, our work is the ﬁrst to pro-
pose an analogue of this assumption to the regression case,
rigorously study it, and consequently derive corresponding
unsupervised ensemble regression schemes.
4. Unsupervised Ensemble Regression
Given only the predictions fi(xj), the simplest unsuper-
vised approach to estimate the response y at an instance x
is to average the m regressors,
ˆyAVG(x) = 1
m
m
X
i=1
fi(x) .
Averaging is the optimal linear estimator when all regres-
sors make independent zero-mean errors of equal variance.
A more robust but non-linear method is the median,
ˆyMED(x) = median
 f1(x), . . . , fm(x)

.
Averaging and median are na¨ıve estimators in the sense that
prediction at each xk depends only on fi(xk) and does not
depend on the other observations fi(xj), xj ̸= xk.
As we show theoretically below and illustrate empirically
in Section 5, under some reasonable assumptions, one can
do signiﬁcantly better than the ensemble mean and median
by analyzing all the data fi(xj) and in particular the m×m
covariance matrix C of the m regressors.
Speciﬁcally, we propose a novel framework to study un-
supervised ensemble regression, based on the assumption
that the m experts make approximately uncorrelated errors
with respect to the optimal predictor. We develop methods
to detect the best and worst regressors and derive U-PCR, a
novel unsupervised principal components ensemble regres-
sor. Similar to Merz and Pazzani (1999), the weight vector
of U-PCR is a linear combination of the top few eigenvec-
tors of C (typically just one or two). The key novelty is that
we estimate the coefﬁcients in a fully unsupervised manner.
To this end, we do assume knowledge of the ﬁrst two mo-
ments of Y . Such knowledge seems inevitable, as other-
wise the observed data may be arbitrarily shifted and scaled
without changing the correlation of the regressors. Know-
ing the ﬁrst moment θ1 = E[Y ], allows to estimate the bias
bi = E[fi(X) −Y ] of each regressor fi by its mean over
the n unlabeled samples,
ˆbi = 1
n
n
X
j=1
fi(xj) −θ1 = bi + oP (1).
Knowledge of Var(Y ) allows a rough estimate of the ac-
curacy of the m regressors. A very accurate regressor must
have Var(fi) ≈Var(Y ), whereas if Var(fi) ≪Var(Y ) or
Var(fi) ≫Var(Y ), then fi must have a large error.
In what follows we consider predicting the mean-centered
responses yj −θ1 by a linear combination of the mean cen-
tered predictors, ˆy(x) = P
j wj(fj(x) −ˆbj −θ1). We thus
work with the mean centered matrix
Zij = fi(xj) −ˆbi −θ1.
This is equivalent to assuming that E[fi(X)] = E[Y ] = 0.
4.1. Statistically Independent Errors
As discussed in Section 2, in light of the optimal weights
in Eq. (6), the key challenge in unsupervised ensemble re-
gression is to estimate the vector ρ of Eq. (4), without any
labeled data.
To this end, we propose the following regression analogue
of the Dawid-Skene assumption of conditionally indepen-
dent experts. Recall that when the risk function is the MSE,
the optimal regressor is the conditional mean,
g(x) = E[Y |X = x].
Its mean is g1 = E[g(X)] = E[Y ] = 0, and its MSE is
E[(Y −g(X))2] = Var(Y ) −g2, where
g2 = EX[g(X)2] = E(X,Y )[g(X)Y ].
(10)

## Page 5

Unsupervised Ensemble Regression
For each regressor, write fi(x) = g(x) + hi(x). Since fi
is mean centered, E[hi(X)] = 0. Hence, ρi = E[fi(X)Y ]
simpliﬁes to
ρi = E[g(X)Y ] + E[hi(X)Y ] = g2 + ai .
(11)
Similarly, the MSE of regressor i is
MSE(fi) = g2 −2ai + E[hi(X)2].
(12)
In this notation, the challenge is thus to estimate g2 and
the vector a = (a1, . . . , am). Inspired by Eq. (9) in the
case of classiﬁcation, we assume the m regressors make
independent errors with respect to g(X), namely that 2
E[hi(X)hj(X)] = 0.
(13)
This assumption is reasonable, for example, when the m
regressors were trained independently and are rich enough
to well approximate the conditional mean g(X). Note that
when the response Y is perfectly predictable from the fea-
tures X, then g(X) = Y and our assumption then states
that the m regressors make independent errors with respect
to the response Y . This can be viewed as the regression
equivalent of the Dawid-Skene model in classiﬁcation.
Next, we consider how to estimate the values ai under the
independent error assumption of Eq. (13). Suppose for a
moment that the value of g2 of Eq. (10) was known. We
shall discuss how to estimate it in the next section. As the
following theorem shows, in this case, we can consistently
estimate ρ by solving a system of linear equations.
Theorem 1. Assume that the given m ≥3 regressors make
pairwise independent errors with respect to the conditional
mean. If g2 is known then given only the data matrix Z, we
can consistently estimate the vector ρ at rate OP (1/√n).
Proof. It is instructive to ﬁrst consider the population set-
ting where n →∞. Here, under the assumption (13), the
off-diagonal entries of the population covariance are
Cij = E[fi(X)fj(X)] = g2 + ai + aj
(14)
Since C is symmetric, these off-diagonal entries pro-
vide
 m
2

linear equations for the m unknown variables
a
=
(a1, . . . , am). Thus, if m ≥3 there are enough
linearly independent equations to uniquely recover a. The
vector ρ can then be computed from Eq. (11).
In practice, the population matrix C is unknown. However,
given the m×n matrix Z, we may estimate it by the sample
covariance ˆC. Since ˆCij = Cij + OP ( 1
√n), estimating
(a1, . . . , am) by least-squares yields a consistent estimator
ˆρ with asymptotic error OP (1/√n).
2Strictly speaking, the assumption is that the deviations from
g(X) are uncorrelated and not necessarily independent.
Remark 1. In practice, assumption (13) that all m regres-
sors make independent errors, may be strongly violated at
least for some pairs. To be robust to deviations from this
assumption one may choose a suitable loss function L(·),
and solve the optimization problem
ˆa = arg min
(a1,...,am)
X
i<j
L( ˆCij −g2 −ai −aj).
(15)
In our experiments, we considered both the absolute loss
and the standard squared loss.
4.2. Unsupervised PCR
The analysis above assumed knowledge of g2, or equiv-
alently of the minimal attainable MSE of the regression
problem at hand. Clearly, this would seldom be known to
the practitioner. Further, any guess of g2 ∈[0, Var(Y )]
gives a valid solution. Speciﬁcally, let ˆa(q) be the solu-
tion of (15) with an assumed value g2 = q. Then, due to
the additive structure inside the parenthesis in Eq. (15), re-
gardless of the loss function L, we have ˆa(q) = ˆa(0) −q
21
where 1 = (1, . . . , 1)T ∈Rm. Similarly, by Eq. (11),
ˆρ(q) = ˆρ(0) + q
2 1 .
(16)
What is needed is thus a model selection criterion that
would be able to accurately estimate the value of g2, given
the family of possible solutions ˆρ(q) for 0 ≤q ≤Var(Y ).
To motivate our proposed estimator of g2, let us ﬁrst ana-
lyze the model of the previous section, but with the addi-
tional assumption that all m regressors are fairly close to
the optimal conditional mean g(x). Namely, for analysis
purposes, we scale the deviations hi by a parameter ǫ,
fi(x) = g(x) + ǫhi(x)
(17)
and study the behaviour of various quantities as a function
of ǫ. Speciﬁcally, under Eq. (17), the population covariance
of the m regressors takes the form
C(ǫ) = g211T + ǫ(a1T + 1aT ) + ǫ2D
where ai = E[hi(X)Y ] and D is a diagonal matrix with
entries Dii = E[h2
i (X)]. The following lemma character-
izes the leading eigenvalue and eigenvector of C, as ǫ →0.
Lemma 2. Let λ1(ǫ), v1(ǫ) be the largest eigenvalue and
corresponding eigenvector of C(ǫ). Then, as ǫ →0,
λ1(ǫ)
=
g2m + (2aT 1) · ǫ + O(ǫ2)
(18)
v1(ǫ)
=
g21 + (a −aT 1
m 1) · ǫ + O(ǫ2).
(19)
Several insights can be gained from this lemma. First, at
ǫ = 0 the matrix C(ǫ = 0) = g211T is rank one with

## Page 6

Unsupervised Ensemble Regression
a single non-zero eigenvector v1 = 1 and corresponding
eigenvalue λ1 = g2m. Hence, if the m regressors are all
very close to g(x), their population matrix C is nearly rank
one and very ill conditioned. Even with an accurate es-
timate of g2 and consequently of ˆρ, inverting Eq. (6) to
estimate ˆw = ( ˆC)−1ˆρ would then be extremely unstable.
Second, under the model (17), ρ = g21 + ǫa. Comparing
this to Eq. (19), the vector ρ and the leading eigenvec-
tor v1, properly scaled, are nearly identical, up to a small
shift by ( 1
m
P ai)ǫ and up to O(ǫ2) terms. Moreover, up
to O(ǫ2) terms, the matrix C(ǫ) has rank two, spanned by
the two vectors 1 and a. Hence, up to O(ǫ2) terms, the
true vector ρ can be written as a linear combination of the
ﬁrst two eigenvectors of C. Thus, even though the ma-
trix C is ill conditioned, a principal component approach,
with just K = 1 or 2 components, can provide an excel-
lent approximation of the optimal weight vector w∗. While
our focus is on unsupervised ensemble, this analysis pro-
vides a rigorous theoretical support for the PCR* method
of Merz and Pazzani (1999), a result which may be of in-
dependent interest for supervised ensemble learning.
Third, since by Eq.(12), MSE(fi) = g2 −2aiǫ + O(ǫ2),
the worst and best regressors may be detected by the largest
and smallest entries in v1 or in the estimated vector ˆρ.
Lemma 2 suggests several ways to estimate the unknown
quantity g2. By Eq. (18), one option is ˆg2 = λ1/m. Under
our assumed model, this would incur an error (2 P aj)ǫ +
O(ǫ2). Another option, which we found works better in
practice is to consider the relation between ˆρ(q) and the top
eigenvector v1 of C, normalized to ∥v1∥= 1. Speciﬁcally,
we estimate g2 by minimizing the following residual,
ˆg2 = arg min
q∈[0,Var(Y )]
RES(q)=arg min
q
∥ˆρ(q) −
 vT
1 ˆρ(q)

v1∥
∥ˆρ(q)∥
(20)
where ˆρ(q) is given in Eq. (16). From the estimate ˆg2, the
weight vector of U-PCR is
wU-PCR = 1
λ1
(vT
1 ˆρ(ˆg2)) v1
(21)
A sketch of our proposed scheme appears in Algorithm 1.
4.3. Practical Issues
Before illustrating the competitive performance of U-PCR,
we discuss several important practical issues that need to
be addressed when handling real-world ensembles, whose
individual regressors may not satisfy our assumptions.
First, when the true value g2 ≪Var(Y ), the regression
problem at hand is very difﬁcult, and no linear combination
of the m predictors can give a small error. If our estimated
ˆg2/Var(Y ) ≤ǫL for some small threshold ǫL, say 0.1, this
Algorithm 1 Sketch of U-PCR
Input: Predictions fi(xj), E[Y ] and Var(Y )
Compute covariance ˆC and its leading eigenvector v1
For q ∈[0, Var(Y )], compute ˆρ(q) by Eqs. (11), (15)
and (16)
Estimate g2 via Eq. (20).
Set ρ = ˆρ(ˆg2) and ρmax = max ρi
if ˆg2 < ǫL · Var(Y ) then
Difﬁcult prediction problem; STOP
end if
Exclude experts with ρi < 0.05 Var(Y ) or ρi < ρmax/3
Recalculate v1, ˆρ(q), ˆg2 on remaining experts
Output: Weight vector ˆw of Eq. (21)
is an indication of such a difﬁcult problem. In this case we
stop and do not attempt to construct an ensemble learner.
Second, even when accurate prediction is possible, in our
experience, if some regressors are far less accurate than
others, then it is important to detect them and exclude them
from the ensemble, and recompute the various quantities
after their removal. However, in the rare cases that after this
removal only m ≤4 regressors remained, then we found it
better to compute their simple average instead of Eq. (21).
Finally, if the second eigenvalue is not extremely small,
then it is beneﬁcial to project the vector ˆρ onto the ﬁrst
two eigenvectors of ˆC . In our experiments we did so when
λ2 > 0.1 · Trace( ˆC). Then, Eq. (21) is replaced by
wU-PCR = 1
λ1
(vT
1 ˆρ(ˆg2)) v1 + 1
λ2
(vT
2 ˆρ(ˆg2)) v2 .
5. Experiments
We illustrate the performance of U-PCR on various real
world regression problems. These include problems for
which we trained multiple regression algorithms, and two
applications where the regressors were constructed by a
third party and only their predictions were given to us.
We compare U-PCR to the ensemble mean and median as
well as to a linear oracle regressor of the form (2), which
has access to all the response values yj. It determines its
weights by ordinary least squares over all n samples
wor = (ZZT )−1Z · (y −θ1) .
(22)
We denote the normalized MSE of the oracle by δor =
MSE(wor)/Var(Y ).
We divide the regression problems into three difﬁculty lev-
els: (i) δor ≲0.1, where accurate prediction is possible by
a linear combination of the m regressors; (ii) 0.1 ≲δor ≲
0.8, a challenging regression task; and (iii) δor ≳0.8, where
the m experts provide very little, if any information on Y .

## Page 7

Unsupervised Ensemble Regression
q / Var(Y)
0
0.2
0.4
0.6
0.8
1
0
0.2
0.4
0.6
0.8
1
MSE
RESIDUAL
MIN RES
λ1/m
(a) CCPP: Accurate prediction possible
q / Var(Y)
0
0.2
0.4
0.6
0.8
1
0
0.2
0.4
0.6
0.8
1
MSE
RESIDUAL
MIN RES
λ1/m
(b) Basketball: Challenging task
q / Var(Y)
0
0.2
0.4
0.6
0.8
1
0
0.5
1
1.5
MSE
RESIDUAL
MIN RES
λ1/m
(c) Affairs:
Limited information on re-
sponse
Figure 1. Plots of MSE(q) and RES(q) for problems of easy, moderate and hard difﬁculty levels. The vertical lines are the estimated ˆg2
at which the residual is minimal and λ1/m.
We start with the following basic question: Given only
fi(xj) and the ﬁrst two moments of Y , can we roughly
estimate the difﬁculty level of our problem? If it belongs to
level (i) or (ii), is it possible to detect the most accurate or
least accurate regressors? Finally, can we construct a linear
combination at least as accurate as the mean or median?
5.1. Manually Crafted Ensembles
With precise details appearing in the supplement, we con-
sidered 18 different prediction tasks, including energy out-
put prediction in a power plant, ﬂight delays, basketball
scoring and more. Each dataset was randomly split into
ntrain samples used to train 10 different regression algo-
rithms and remaining n samples to construct the observa-
tions fi(xj), see Table 3 in Supplementary. The regressors
included Ridge Regression, SVR, Kernel Regression and
Decision Trees, among others.
Table 1 in the supplement shows the MSE of U-PCR, mean
and median averaged over 20 repetitions, each with differ-
ent random splits into train and test samples. On several
datasets, U-PCR obtained a signiﬁcantly lower MSE. With
further details in the supplement, here we highlight some
of our key results. We start by estimating g2 and classi-
fying the problems by difﬁculty level. Fig. 1 shows this
estimation procedure on three datasets. The x-axis is the
value of q normalized by Var(Y ). The black curve is the
unobserved MSE(q) obtained by the weight vector of Eq.
(21), with assumed ˆρ(q). The red curve is the computed
residual RES(q) of Eq. (20) and the vertical line is the es-
timated ˆg2. Our approach is indeed able to correctly detect
the difﬁculty levels of these problems and estimate a value
ˆg2, whose corresponding MSE is not too far from the min-
imal achievable by using any of the ˆρ(q). Fig. 7 in the
supplement shows the estimated ˆρ vs. the true ρ. For easy
problems with g2 ≪Var(Y ) the agreement is remarkable.
Next, we evaluated the ability to detect the most accurate
regressor in the ensemble. We measured the excess risk of
selecting the regressor with the smallest estimated MSE,
compared to the best regressor, which is unknown. Addi-
tionally, we measured the excess risk in selecting the single
regressor with the greatest corresponding entry in the lead-
ing eigenvector of C. The details are given in Table 2 of the
Supplementary. Our experiments show that in most cases
choosing the predictor with lowest estimated MSE outper-
forms the one with largest entry in v1.
Fig. 2 shows the effectiveness of detecting inaccurate re-
gressors, by pruning those whose entries ˆρi < ρmax/3 or
ˆρi < 0.05Var(Y ). Finally, Fig. 3 illustrates the advantages
of U-PCR over the mean and median, on problems of easy
to moderate difﬁculty.
5.2. HPN-DREAM Challenge Experiment
Next, we consider real world problems where the ensem-
ble regressors were constructed by a third party. The ﬁrst
problem came from the HPN-DREAM breast cancer net-
work inference challenge (Hill et al., 2016a). Here, partici-
pants were asked to predict the time varying concentrations
of 4 proteins after the introduction of an inhibitor. We were
given the predictions of m = 12 models on n ≈2500 in-
stances. We constructed a separate U-PCR model for each
protein. Fig. 2 demonstrates the success of our method
in detecting accurate regressors and removing inaccurate
ones. Fig. 4 shows that U-PCR outperformed the mean
and median on 3 of the 4 proteins. We note that for all
four proteins, the single best model had comparable MSE
to U-PCR, however, this model is unknown. For three of
the four proteins U-PCR had smaller MSE than that of the
single model estimated as being the most accurate.
5.3. Bounding Box Experiment
Here we were given the predictions of 6 deep learning mod-
els trained by Seematics Inc., on the location of physi-
cal objects in images.
The models were trained on the
PASCAL Visual Object Classes dataset (Everingham et al.,

## Page 8

Unsupervised Ensemble Regression
TRUE MSE
0.4
0.5
0.6
0.7
0.8
0.9
1
ESTIMATED MSE
0.4
0.5
0.6
0.7
0.8
0.9
1
1.1
1.2
OUTLIERS
EST BEST
(a) Before outlier removal
TRUE MSE
0.4
0.5
0.6
0.7
0.8
0.9
1
ESTIMATED MSE
0.4
0.5
0.6
0.7
0.8
0.9
1
1.1
1.2
EST BEST
(b) After outlier removal
TRUE MSE
0
0.5
1
1.5
2
2.5
ESTIMATED MSE
0
0.5
1
1.5
2
2.5
3
OUTLIERS
EST BEST
(c) Before outlier removal
TRUE MSE
0
0.2
0.4
0.6
0.8
1
ESTIMATED MSE
0
0.2
0.4
0.6
0.8
1
EST BEST
(d) After outlier removal
Figure 2. Estimated MSE(fi) vs. true MSE for Flights AUS (left panels) and UACC812 protein (right panels) before and after outlier
removal. The outlier removal scheme is not based on the estimated MSE, but rather as described in main text, on the entries of the
estimated ˆρ. In some datasets, such as Flights AUS, recalculation after this removal gives more accurate estimates of regressors’ MSE.
MSE(oracle)/Var(Y)
0
0.02
0.04
0.06
0.08
0.1
EXCESS RISK
0
0.05
0.1
0.15
0.2
MEAN
MED
UPCR
(a) Nearly Perfect Prediction
MSE(oracle) / Var(Y)
0.2
0.3
0.4
0.5
0.6
0.7
EXCESS RISK
0
0.1
0.2
0.3
0.4
0.5
MEAN
MED
UPCR
(b) Challenging Problems
Figure 3. Excess risk MSE(ensemble)−MSE(oracle), divided by Var(Y ) for easy problems (left) and challenging ones (right).
MSE(oracle) / Var(Y)
0.1
0.15
0.2
0.25
0.3
0.35
EXCESS RISK
0
0.05
0.1
0.15
0.2
MEAN
MED
UPCR
Figure 4. HPN-DREAM Challenge Accuracy.
2012), whereas the predictions were made on images from
COCO dataset (Lin et al., 2014). We focused on three ob-
ject classes {person, dog, cat}, with each neural network
providing four coordinates for the bounding box: (x1, y1)
and (x2, y2). We used U-PCR as an ensemble predictor
each coordinate separately, with the mean squared error as
out measure of accuracy. An example of the MSE estima-
tion by our method can be seen in Fig. 5, and the accuracy
for all object classes and all coordinates in Fig. 6. Results
on few images are in the supplementary.
6. Summary and Discussion
In this paper we tackled the problem of unsupervised en-
semble regression. We presented a framework to explore
this problem, based on an independent error assumption.
We proposed methods, together with theoretical support, to
TRUE MSE
0.4
0.6
0.8
1
ESTIMATED MSE
0
0.5
1
EST BEST
Figure 5. MSE estimation for class Cat, coordinate x1
MSE(oracle)/Var(Y)
0.05
0.1
0.15
EXCESS RISK
0
0.02
0.04
0.06
0.08
MEAN
MED
UPCR
Figure 6. Bounding box prediction accuracies
detect the best and worst regressors and to linearly aggre-
gate them, all in an unsupervised manner. As our theoret-
ical analysis in Section 4 showed, unsupervised ensemble
regression is different from the well studied problem of un-
supervised ensemble classiﬁcation, and required different
approaches to its solution.
Our work raises several questions. One of them is how to

## Page 9

Unsupervised Ensemble Regression
extend our method to a semi-supervised setting, in which
there is also a limited amount of labeled data. It is also
interesting to theoretically understand the relative beneﬁts
of labeled versus unlabeled data for ensemble learning.
Another direction for future research is to replace the strict
independent error assumption by more complicated yet re-
alistic models for dependencies between the regressors.
In the context of unsupervised classiﬁcation, Fetaya et al.
(2016) relaxed the conditional independence model of
Dawid and Skene by introducing an intermediate layer of
latent variables. Instead of a rank-one off diagonal covari-
ance, the matrix C in their model had a low rank structure,
which the authors learned by a spectral method. It is inter-
esting whether a similar approach can be developed for an
ensemble of regressors.
Acknowledgments
The authors thank Moshe Guttmann and the Seematics
team for their help. This research was funded by the In-
tel Collaborative Research Institute for Computational In-
telligence (B.N.) and by NIH grant 1R01HG008383-01A1
(Y.K. and B.N.).
References
Anandkumar, A., Ge, R., Hsu, D., Kakade, S. M., and Tel-
garsky, M. (2014). Tensor decompositions for learning
latent variable models. Journal of Machine Learning Re-
search, 15(1):2773–2832.
Breiman, L. (1996). Stacked regressions. Machine Learn-
ing, 24(1):49–64.
Breiman, L. (2001). Random forests. Machine learning,
45(1):5–32.
Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J.
(2009). Modeling wine preferences by data mining from
physicochemical properties. Decision Support Systems,
47(4):547–553.
Dawid, A. P. and Skene, A. M. (1979). Maximum like-
lihood estimation of observer error-rates using the em
algorithm. Applied statistics, pages 20–28.
Donmez, P., Lebanon, G., and Balasubramanian, K. (2010).
Unsupervised supervised learning i: Estimating classiﬁ-
cation and regression errors without labels. Journal of
Machine Learning Research, 11:1323–1351.
Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J.,
and Zisserman, A. (2012). The PASCAL Visual Object
Classes Challenge 2012 (VOC2012) Results.
Fanaee-T, H. and Gama, J. (2014). Event labeling com-
bining ensemble detectors and background knowledge.
Progress in Artiﬁcial Intelligence, 2(2-3):113–127.
Fetaya, E., Nadler, B., Jaffe, A., Kluger, Y., and Jiang, T.
(2016). Unsupervised ensemble learning with dependent
classiﬁers. In AISTATS, pages 351–360.
Freund, Y. and Schapire, R. E. (1995). A decision-theoretic
generalization of on-line learning and an application to
boosting.
In European conference on computational
learning theory, pages 23–37.
Friedman, J., Hastie, T., and Tibshirani, R. (2000). Ad-
ditive logistic regression: a statistical view of boosting.
The Annals of Statistics, 28(2):337–407.
Friedman, J. H. (1991). Multivariate adaptive regression
splines. The annals of statistics, pages 1–67.
Hill, S. M., Heiser, L. M., Cokelaer, T., Unger, M., Nesser,
N. K., Carlin, D. E., Zhang, Y., Sokolov, A., Paull, E. O.,
Wong, C. K., et al. (2016a). Inferring causal molecular
networks: empirical assessment through a community-
based effort. Nature methods, 13(4):310–318.
Hill, S. M., Nesser, N. K., Johnson-Camacho, K., Jef-
fress, M., Johnson, A., Boniface, C., Spencer, S. E., Lu,
Y., Heiser, L. M., Lawrence, Y., et al. (2016b). Con-
text speciﬁcity in causal signaling networks revealed by
phosphoprotein proﬁling. Cell systems.
Ionita-Laza, I., McCallum, K., Xu, B., and Buxbaum, J. D.
(2016). A spectral approach integrating functional ge-
nomic annotations for coding and noncoding variants.
Nature genetics, 48(2):214–220.
Jaffe, A., Nadler, B., and Kluger, Y. (2015). Estimating the
accuracies of multiple classiﬁers without labeled data. In
AISTATS.
Johnson, V. E. (1996). On Bayesian Analysis of Multi-
rater Ordinal Data: An Application to Automated Essay
Grading. Journal of the American Statistical Associa-
tion, 91(433):42–51.
Jolliffe, I. (2002). Principal component analysis. John Wi-
ley & Sons, Ltd.
Kato, T. (1995). Perturbation Theory of Linear Operators.
Springer, Berlin, second edition.
Lavancier, F. and Rochet, P. (2016). A general procedure
to combine estimators. Computational Statistics & Data
Analysis, 94:175–192.
Leblanc, M. and Tibshirani, R. (1996). Combining Esti-
mates in Regression and Classiﬁcation. Journal of the
American Statistical Association, 91(436):1641–1650.

## Page 10

Unsupervised Ensemble Regression
Lichman, M. (2013). UCI machine learning repository.
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P.,
Ramanan, D., Doll´ar, P., and Zitnick, C. L. (2014). Mi-
crosoft coco: Common objects in context. In European
Conference on Computer Vision, pages 740–755.
Mendes-Moreira, J., Soares, C., Jorge, A. M., and Sousa,
J. F. D. (2012). Ensemble approaches for regression: A
survey. ACM Computing Surveys (CSUR), 45(1):10.
Merz, C. and Pazzani, M. (1999). A Principal Components
Approach to Combining Regression Estimates. Machine
Learning, 36(1-2):9–32.
Nadler, B. (2008). Finite sample approximation results for
principal component analysis: A matrix perturbation ap-
proach. The Annals of Statistics, pages 2791–2817.
Parisi, F., Strino, F., Nadler, B., and Kluger, Y. (2014).
Ranking and combining multiple predictors without la-
beled data. Proceedings of the National Academy of Sci-
ences of the United States of America, 111(4):1253–8.
Perrone, M. P. and Cooper, L. N. (1992). When networks
disagree: Ensemble methods for hybrid neural networks.
World Scientiﬁc.
Platanios, E. A., Blum, A., and Mitchell, T. (2014). Esti-
mating accuracy from unlabeled data. In Uncertainty in
Artiﬁcial Intelligence, pages 682–691.
Platanios, E. A., Dubey, A., and Mitchell, T. (2016). Es-
timating accuracy from unlabeled data: A bayesian ap-
proach. In International Conference on Machine Learn-
ing, pages 1416–1425.
Raykar, V. C., Yu, S., Zhao, L. H., Valadez, G. H.,
Florin, C., Bogoni, L., and Moy, L. (2010). Learning
from crowds. Journal of Machine Learning Research,
11(Apr):1297–1322.
Sheng, V. S., Provost, F., and Ipeirotis, P. G. (2008). Get
another label? improving data quality and data mining
using multiple, noisy labelers. In Proceedings of the 14th
ACM SIGKDD international conference on Knowledge
discovery and data mining, pages 614–622. ACM.
Timmermann, A. (2006). Forecast combinations. Hand-
book of economic forecasting, 1:135–196.
Whitehill, J., Wu, T.-f., Bergsma, J., Movellan, J. R., and
Ruvolo, P. L. (2009). Whose vote should count more:
Optimal integration of labels from labelers of unknown
expertise. In Advances in neural information processing
systems, pages 2035–2043.
Wolpert, D. H. (1992). Stacked generalization. Neural net-
works, 5(2):241–259.
Wu, D., Lawhern, V. J., Gordon, S., Lance, B. J., and
Lin, C.-T. (2016).
Spectral meta-learner for regres-
sion (smlr) model aggregation: Towards calibrationless
brain-computer interface. In IEEE International Confer-
ence on Systems, Man, and Cybernetics, pages 743–749.
Zhang, Y., Chen, X., Zhou, D., and Jordan, M. I. (2014).
Spectral methods meet em: A provably optimal algo-
rithm for crowdsourcing. In Advances in neural infor-
mation processing systems, pages 1260–1268.
Zhou, D., Basu, S., Mao, Y., and Platt, J. C. (2012). Learn-
ing from the wisdom of crowds by minimax entropy.
In Advances in Neural Information Processing Systems,
pages 2195–2203.

## Page 11

Unsupervised Ensemble Regression
Supplementary Material
A. Proofs
Proof of Lemma 2. The proof follows a perturbation approach similar to the one outlined in Nadler (2008). Since C(ǫ) is
symmetric and quadratic in ǫ, classical results on perturbation theory (Kato, 1995) imply that in a small neighborhood of
ǫ = 0, the leading eigenvalue and eigenvector are analytic in ǫ. We may thus expand them in a Taylor series,
λ(ǫ)
=
λ0 + λ1ǫ + λ2ǫ2 + . . .
v(ǫ)
=
v0 + v1ǫ + v2ǫ2 + . . .
We insert this expansion into the eigenvector equation C(ǫ)v(ǫ) = λ(ǫ)v(ǫ) and solve the resulting equations at increasing
powers of ǫ.
The leading order equation reads g211T v0 = λ0v0, which gives v0 ∝1 and λ0 = g2∥1∥2 = g2m. Since the eigenvector
v(ǫ) is deﬁned only up to a multiplicative factor, we conveniently chose it to be that 1T v(ǫ) = g2m holds for all ǫ. This
gives v0 = g21 and vT
1 v0 = 0.
The O(ǫ) equation reads
g211Tv1 + (a1T + 1aT )v0 = λ0v1 + λ1v0.
(23)
Multiplying this equation from the left by vT
0 gives
2(vT
0 1)(aT v0) = λ1∥v0∥2
or λ1 = 2 P aj. Thus, Eq. (18) follows. Inserting the expression for λ1 back into Eq. (23) gives
v1 = 1
λ0
[(aT v0)1 + (1T v0)a −(2
X
j
aj)v0]
from which Eq. (19) readily follows.
B. Datasets & Results
B.1. Selecting a Single Regressor
Table 2 compares the MSE of the single regressor estimated to be the most accurate, versus the MSE of the single best
regressor, which is unknown in this setup. The following two methods were compared: (i) Selecting the regressor with the
maximum entry ˆρi, and (ii) selecting the regressor with the minimal estimated MSE. The experiments were repeated 20
times for each dataset, mean and standard deviations are reported. All values are normalized by Var(Y ) for fair comparison.
B.2. Dataset Descriptions
Below is a list of the prediction tasks for which we manually trained ensembles with 10 regressors. Table 3 summarizes the
main characteristics of each dataset, and Table 1 contains the mean squared errors of the different approaches normalized
by Var(Y ). The experiments were repeated 20 times, and the mean and standard deviations are reported. We used standard
Python packages for the regression algorithms with the following parameters: Ridge (α = 0.5), Kernel Regression (kernel
chosen using cross validation between polynomial, RBF, sigmoid), Lasso (α = 0.1), Orthogonal Matching Pursuit, Linear
SVR (C = 1), SVR with RBF kernel (C chosen using cross validation out of 0.01, 0.1, 1, 10), Regression Tree (depth 4),
Regression Tree (inﬁnite depth), Random Forest (100 trees), and a Bagging Regressor.
Abalone.
A dataset containing features of abalone, where the goal is to predict its age (Lichman, 2013).
archive.ics.uci.edu/ml/datasets/Abalone
Affairs.
A dataset containing features describing an individual such as time at work, time spent with spouse,
and time spent with a paramour.
The goal here is to predict the time spent in extramarital affairs.
statsmod-
els.sourceforge.net/0.6.0/datasets/generated/fair.html

## Page 12

Unsupervised Ensemble Regression
Table 1. Mean squared error of different ensemble methods, normalized by Var(Y ). On the Affairs data U-PCR estimates it is a difﬁcult
problem and does not predict outcomes. Numbers in bold represent cases where one of the unsupervised ensemble regressors was
signiﬁcantly better than the others.
DATASET
ORACLE
U-PCR
MEAN
MEDIAN
ABALONE
0.43 (±0.01)
0.49 (±0.01)
0.49 (±0.01)
0.49 (±0.01)
AFFAIRS
0.92 (±0.00)
N.A.
0.96 (±0.01)
0.94 (±0.00)
BASKETBALL
0.28 (±0.01)
0.35 (±0.01)
0.35 (±0.00)
0.36 (±0.00)
BIKE SHARING
0.00 (±0.00)
0.00 (±0.00)
0.02 (±0.00)
0.00 (±0.00)
BLOG FEEDBACK
0.41 (±0.03)
0.49 (±0.02)
0.50 (±0.02)
0.58 (±0.02)
CCPP
0.06 (±0.00)
0.07 (±0.00)
0.07 (±0.00)
0.07 (±0.00)
FLIGHTS AUS
0.33 (±0.04)
0.46 (±0.07)
0.58 (±0.06)
0.66 (±0.08)
FLIGHTS BOS
0.47 (±0.04)
0.58 (±0.08)
0.66 (±0.03)
0.69 (±0.08)
FLIGHTS BWI
0.44 (±0.06)
0.56 (±0.09)
0.71 (±0.03)
0.82 (±0.08)
FLIGHTS HOU
0.40 (±0.09)
0.59 (±0.07)
0.69 (±0.03)
0.75 (±0.08)
FLIGHTS JFK
0.50 (±0.05)
0.78 (±0.21)
0.74 (±0.03)
0.90 (±0.04)
FLIGHTS LGA
0.47 (±0.04)
0.59 (±0.06)
0.70 (±0.03)
0.78 (±0.09)
FLIGHTS LONGHAUL
0.69 (±0.05)
0.89 (±0.24)
0.86 (±0.06)
0.97 (±0.01)
FRIEDMAN1
0.02 (±0.00)
0.13 (±0.00)
0.18 (±0.01)
0.16 (±0.01)
FRIEDMAN2
0.00 (±0.00)
0.06 (±0.01)
0.08 (±0.01)
0.07 (±0.01)
FRIEDMAN3
0.04 (±0.01)
0.09 (±0.02)
0.20 (±0.02)
0.22 (±0.02)
ONLINE VIDEOS
0.09 (±0.01)
0.18 (±0.01)
0.22 (±0.01)
0.28 (±0.02)
WINE QUALITY WHITE
0.60 (±0.01)
0.64 (±0.01)
0.66 (±0.01)
0.69 (±0.01)
RHO TRUE / Var(Y)
0.7
0.8
0.9
1
RHO EST / Var(Y)
0.7
0.75
0.8
0.85
0.9
0.95
(a) CCPP: Accurate prediction possible
RHO TRUE / Var(Y)
0.4
0.5
0.6
0.7
0.8
RHO EST / Var(Y)
0.3
0.4
0.5
0.6
0.7
0.8
(b) Basketball: Challenging task
RHO TRUE / Var(Y)
0.04
0.05
0.06
0.07
0.08
0.09
RHO EST / Var(Y)
0.05
0.1
0.15
(c) Affairs: No information on response
Figure 7. Estimated ˆρ vs. true ρ in three regression problems of different difﬁculty levels
Basketball.
Dataset contains stats on NBA players. Task: Predict number of points scored by the player on the next
game. The features are: name, venue, team, date, start, pts ma, min ma, pts ma 1, min ma 1, pts, where
start is whether or not the player started, pts is number of points scored, min is number of minutes played, ma stands
for moving average, starts at season, and ma 1 is a moving average with a 1 game lag.
Bike
Sharing.
Bike
sharing
service
statistics,
including
weather
and
seasonal
information
(Fanaee-T and Gama,
2014).
The
prediction
task
here
is
the
daily
and
hourly
count of
bikes
rented.
archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
Blog Feedback.
Instances in this dataset contain features extracted from blog posts. The task associated with the data is
to predict how many comments the post will receive.
Flights.
Information on ﬂights from 2008, where the task is to predict the delay upon arrival in minutes. The features
here are the date, day of the week, scheduled and actual departure times, scheduled arrival times, ﬂight ID, tail number,
origin, destination, and distance. Due to its size, we split this dataset to ﬂights originating from speciﬁc airports (AUS,
BOS, BWI, HOU, JFK, and LGA), and long-haul ﬂights. stat-computing.org/dataexpo/2009/the-data.html

## Page 13

Unsupervised Ensemble Regression
Table 2. MSE of the single best estimated regressor
DATASET
ORACLE MSE
BEST REGRESSOR MSE
MSE OF arg mini \
MSEi
MSE OF arg maxi ˆρi
ABALONE
0.43 (±0.01)
0.45 (±0.01)
0.49 (±0.03)
0.77 (±0.21)
BASKETBALL
0.28 (±0.01)
0.32 (±0.01)
0.36 (±0.01)
0.49 (±0.13)
BIKE SHARING
0.00 (±0.00)
0.00 (±0.00)
0.00 (±0.00)
0.00 (±0.00)
BLOG FEEDBACK
0.41 (±0.03)
0.43 (±0.03)
0.66 (±0.02)
0.62 (±0.19)
CCPP
0.06 (±0.00)
0.07 (±0.00)
0.07 (±0.00)
0.09 (±0.02)
FLIGHTS AUS
0.33 (±0.04)
0.48 (±0.03)
0.56 (±0.09)
0.66 (±0.24)
FLIGHTS BOS
0.47 (±0.04)
0.53 (±0.04)
0.61 (±0.13)
1.04 (±0.19)
FLIGHTS BWI
0.44 (±0.06)
0.50 (±0.05)
0.50 (±0.05)
0.87 (±0.45)
FLIGHTS HOU
0.40 (±0.09)
0.52 (±0.06)
0.53 (±0.08)
1.13 (±0.45)
FLIGHTS JFK
0.50 (±0.05)
0.54 (±0.05)
0.64 (±0.16)
0.95 (±0.22)
FLIGHTS LGA
0.47 (±0.04)
0.53 (±0.03)
0.59 (±0.12)
1.04 (±0.42)
FLIGHTS LONGHAUL
0.69 (±0.05)
0.74 (±0.05)
0.79 (±0.11)
1.17 (±1.04)
FRIEDMAN1
0.02 (±0.00)
0.03 (±0.00)
0.24 (±0.04)
0.03 (±0.00)
FRIEDMAN2
0.00 (±0.00)
0.01 (±0.00)
0.14 (±0.02)
0.02 (±0.03)
FRIEDMAN3
0.04 (±0.01)
0.07 (±0.01)
0.25 (±0.21)
0.14 (±0.04)
ONLINE VIDEOS
0.09 (±0.01)
0.10 (±0.01)
0.34 (±0.02)
0.17 (±0.03)
WINE QUALITY WHITE
0.60 (±0.01)
0.62 (±0.01)
0.77 (±0.01)
1.12 (±0.19)
Table 3. PREDICTION PROBLEMS
Name
n
nTRAIN
d
MSE(f)
mini MSE(fi)
MSEORACLE
ABALONE
3277
700
7
0.59
0.45
0.431 (±0.006)
AFFAIRS
5466
700
7
1.08
0.93
0.922 (±0.004)
BASKETBALL
48899
900
9
0.43
0.32
0.281 (±0.005)
BIKE SHARING
15579
1600
16
0.07
0.00
0.000 (±0.000)
BLOG FEEDBACK
24197
28000
280
0.64
0.43
0.415 (±0.026)
CCPP
8968
400
4
0.10
0.07
0.059 (±0.001)
FLIGHTS AUS
47595
1000
10
0.76
0.48
0.329 (±0.035)
FLIGHTS BOS
112705
1000
10
0.84
0.53
0.470 (±0.042)
FLIGHTS BWI
101665
1000
10
0.85
0.50
0.440 (±0.065)
FLIGHTS HOU
53044
1000
10
0.87
0.52
0.397 (±0.094)
FLIGHTS JFK
113960
1000
10
0.89
0.54
0.495 (±0.051)
FLIGHTS LGA
111911
1000
10
0.86
0.53
0.471 (±0.040)
FLIGHTS LONG HAUL
9393
1000
10
1.00
0.73
0.686 (±0.051)
FRIEDMAN1
18800
1000
10
0.31
0.03
0.024 (±0.001)
FRIEDMAN2
19400
400
4
0.17
0.01
0.004 (±0.001)
FRIEDMAN3
19400
400
4
0.35
0.07
0.043 (±0.006)
ONLINE VIDEOS
66484
2100
21
0.34
0.10
0.094 (±0.006)
WINE QUALITY WHITE
3598
1100
11
0.79
0.62
0.595 (±0.011)
n is the number of held-out samples. The input X is d dimensional, and the same ntrain random samples were
used to train the different algorithms in the ensemble. MSE(f) is the average regressor error, mini MSE(fi)
is the minimal error achieved by a regressor in the ensemble, and MSEORACLE is the MSE of the oracle,
normalized by Var(Y ), with its standard deviation in parenthesis. For each dataset the split between train
and test was performed 20 times, averages are listed.
CCPP.
Combined
Cycle
Power
Plant
UCI-dataset
containing
physical
characteristics
such
as
tempera-
ture and humidity.
The task here is to predict the net hourly electrical energy output of the plant.
archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
Friedman #1
Motivated by Breiman (1996), we used simulated data according to Friedman (1991). The predictor
variables x1, . . . , x5 are independent and uniformly distributed over [0, 1]. The response is
y = 10 sin(πx1x2) + 20(x3 −.5)2 + 10x4 + 5x5 + ǫ

## Page 14

Unsupervised Ensemble Regression
(a)
(b)
(c)
(d)
(e)
(f)
Figure 8. Sample images from the bounding box experiment. The ground truth bounding box is shown in blue, U-PCR in dashed green,
and the regressors are shown in red.
and ǫ ∼N(0, 1).
Friedman #2
The second data set tested by Friedman (1991) simulated impedance in an alternating current circuit.
Here four predictor variables x1, . . . , x4 are uniformly distributed over the ranges [0, 100], [40π, 560π], [0, 1] and [1, 11]
respectively. The response was
y =
q
x2
1 + (x2x3 −(1/x2x4))2 + ǫ2
with ǫ2 ∼N(0, σ2
2), where the variance was chosen to provide a 3-to-1 signal to noise ratio. For the third dataset in this
series Friedman #3, see the original paper (Friedman, 1991).
Online Videos.
YouTube video transcoding dataset. Predict the transcoding time based on parameters of the video.
archive.ics.uci.edu/ml/datasets/Online+Video+ Characteristics+and+Transcoding+Time+Dataset
Wine Quality White.
Predict the quality score (1-10) of white wine based on chemical characteristics, such as acidity
and pH level (Cortez et al., 2009). archive.ics.uci.edu/ml/datasets/Wine+Quality
