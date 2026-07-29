---
source_pdf: papers/Online Auditing of Information Flow.pdf
slug: online-auditing-of-information-flow
pages: 24
extracted_on: 2026-07-29
---

# Online Auditing of Information Flow

## Page 1

arXiv:2310.14595v1  [cs.LG]  23 Oct 2023
Online Auditing of Information Flow
Mor Oren-Loberman
Vered Azar
Wasim Huleihel
October 24, 2023
Abstract
Modern social media platforms play an important role in facilitating rapid dis-
semination of information through their massive user networks. Fake news, misinfor-
mation, and unveriﬁable facts on social media platforms propagate disharmony and
aﬀect society. In this paper, we consider the problem of online auditing of information
ﬂow/propagation with the goal of classifying news items as fake or genuine. Speciﬁ-
cally, driven by experiential studies on real-world social media platforms, we propose a
probabilistic Markovian information spread model over networks modeled by graphs.
We then formulate our inference task as a certain sequential detection problem with
the goal of minimizing the combination of the error probability and the time it takes
to achieve correct decision. For this model, we ﬁnd the optimal detection algorithm
minimizing the aforementioned risk and prove several statistical guarantees. We then
test our algorithm over real-world datasets. To that end, we ﬁrst construct an oﬄine
algorithm for learning the probabilistic information spreading model, and then apply
our optimal detection algorithm. Experimental study show that our algorithm outper-
forms state-of-the-art misinformation detection algorithms in terms of accuracy and
detection time.
1
Introduction
Modern social media platforms signiﬁcantly facilitate the rapid dissemination of information
through their massive user networks. Recent surveys indicate that 73% of people receive news
from social media [14], and as many as 72% of adult Internet users in the U.S. have used social
network sites for health related advice, the majority of which following experiences shared by
friends on social media. The ease of posting and sharing news, coupled with recent advances
in AI technology, expedite the propagation of rumors, misinformation, and even maliciously
fake information. Unfortunately, the ability of AI algorithms to identify such items grows
M. Oren-Loberman, V. Azar and W. Huleihel are with the Department of Electrical Engineering-
Systems
at
Tel
Aviv
university,
Tel
Aviv
6997801,
Israel
(e-mails:
orenmor@mail.tau.ac.il,
vered.azr@gmail.com, wasimh@tauex.tau.ac.il). This work is supported by the ISRAEL SCIENCE
FOUNDATION (grant No. 1734/21).
1

## Page 2

more slowly than the ability to create it [11]. It is therefore of major importance to develop
a better theoretical understanding of the social structure that enables the propagation of
such items, to guide the development of robust methodologies and eﬃcient countermeasures
to tackle this problem.
Recently, it was shown empirically in [16] that in many social networks falsehood informa-
tion/misinformation/rumors diﬀuse signiﬁcantly farther, faster, deeper, and more broadly
than the truth, in all categories of information, and in many cases by an order of magnitude.
Moreover, it was observed that the spread of fake information is essentially not due to social
bots that are programmed to disseminate inaccurate stories. Instead, fake news speeds faster
around, say, Twitter, due to people retweeting inaccurate news items. This observation was
later used in [10] to provide a heuristic recovery algorithm for fake news detection using
geometric deep learning.
Despite the fact that the topic of automatic misinformation detection received a signif-
icant attention in the literature, it is still considered as a challenging daunting task. As
an initial approach of combating the spread of misinformation, many social media platforms
exploit their massive crowd and resources to employ “human-based” methods, such as crowd
sourcing user feedbacks and third-party fact-checking. Even though these methods have the
potential of achieving high accuracy rate, they are most often unscalable and signiﬁcantly
slow. In order to cope with the crucial drawbacks of these primitive methods, signiﬁcant
work has gone into research on automatic misinformation detection in a fast, scaleable and
accurate manner. It has been demonstrated in [1] that utilizing users and posts content
features for misinformation detection can be very eﬀective. This observation drives many
machine learning, data-mining, and AI approaches to automatically detect misinformation
using feature extraction (see, a recent survey in [15]).
Perhaps surprisingly, in spite of the crucial importance of quick misinformation detection
due to its deceptive nature, this aspect is still in its early stage of development. Several early
misinformation detection algorithms that aim to debunk rumors at their stage of diﬀusion
have been developed [2, 5–9, 13, 19], yet, these do not make a real-time decision and require
a pre-determined number of observations as input. In this paper, our goal is to provide a
systematic theoretic investigation of the interesting ﬁndings in [16]. In particular, we would
like to understand and answer the following meta-question:
Is it possible to infer whether pieces of information propagated over time in a social net-
work are falsehood or truth based only on the way they diﬀuse over the network?
An intriguing framework was suggested in [17] for real-time quickest misinformation de-
tection, using a Markov optimal stopping problem, based on a probabilistic information
spreading model.
Under this framework, the authors propose a data-driven and model-
driven algorithm, termed QuickStop, for real-time misinformation detection. This algorithm
consists of an oﬄine procedure for learning the probabilistic information spreading model,
and an online algorithm to detect misinformation. Our paper is greatly inspired by [17],
however, we propose and analyze a more general and realistic diﬀusion model, that takes
into account several practical aspects, such as the social network graph structure, and the
possibility of missing data, as we explain below in detail.
2

## Page 3

Speciﬁcally, to approach the question above, we introduce a Markovian information
spreading model over a social network modeled by a graph.
When a complete network
and information diﬀusion information are known, the information spreading trace is likely
to be a tree or a forest (when multiple information sources exist). However, in practice, it is
often not the case because of missing information and partial observations, see, e.g., [3, 4].
Accordingly, in our model we assume that only arbitrary parts of the information spread-
ing traces are observed by the auditor. Thus, while the underlying information traces are
Markovian, the actual observed information behaves as a more complicated hidden Markov
chain. With this model, we deﬁne the auditor’s goal using a sequential hypothesis testing
problem, where under the null hypothesis the underlying information is genuine, and under
the alternative hypothesis it is fake. We show that this problem can be formulated as a
certain optimal stopping problem.
Using this formulation, we derive an optimal procedure for real-time, quickest cost-
eﬃcient misinformation detection, with a straightforward stopping policy. We analyze the
performance of this algorithm, by driving bounds on the associated conditional error prob-
abilities, using an equivalent sequential probability ratio test (SQRT) of our detection algo-
rithm. Finally, we test our algorithm over real-world datasets. To that end, we construct an
oﬄine algorithm for learning the probabilistic information spreading model, and then apply
our optimal detection algorithm. Our experimental results on real-world dataset show that
our algorithm outperforms state-of-the-art misinformation detection algorithms in terms of
accuracy and detection time.
Notation.
We use calligraphic font to indicate sets, and sans serif font with uppercase and
lowercase letters X, x to indicate RVs and their values, respectively. P(·) and E [·] indicate
the probability and expectation functions. 1E is the indicator function that gets 1 when an
event E is true and 0 otherwise. We denote the cardinality of some set S by |S|. For a set
X , we let X n denote the n-fold Cartesian product of X . An element of X n is denoted by
xn = (x1, x2, . . . , xn). A substring of xn ∈X n is designated by xj
i = (xi, xi+1, . . . , xj), for
1 ≤i ≤j ≤n; when i = 1, the subscript is omitted. A directed walk is a ﬁnite or inﬁnite
sequence of edges directed in the same direction which joins a sequence of vertices. Let
G = (V, E) be a directed graph. A ﬁnite directed walk is a sequence of edges e1, e2, . . . , en−1
for which there is an associated sequence of vertices (v1, v2, . . . , vn) such that ei = (vi, vi+1),
for i = 1, 2, . . . , n −1. The sequence (v1, v2, . . . , vn) is the vertex sequence of the directed
walk. A directed path is a directed walk in which all vertices and edges are distinct.
2
Problem Formulation
Underlying graph.
Consider an online social network platform that is monitoring the
spread of some information in the network. Let G = (V, E) be a directed graph representing
a social network platform that is monitoring the spread of some information, where V = [n]
is the set of nodes (e.g., users), and E is the set of directed edges (e.g., connections between
users). Each node u ∈V is associated with a d-dimensional feature vector xu ∈Rd. We
assume two types of information, either genuine or fake, can spread over G; we denote its
type by I ∈{0, 1}, with I = 0 refers to a genuine information, and I = 1 refers to a fake one.
3

## Page 4

0 0.10.20.30.40.50.60.70.80.9 1
0
0.2
0.4
0.6
0.8
Score
Frequency
(a) Genuine/real news
0 0.10.20.30.40.50.60.70.80.9 1
0
0.2
0.4
0.6
Score
Frequency
(b) Fake news
Figure 1: Distributions of linear SVM classiﬁcation scores, associated with the edge-based
model, over the Weibo dataset.
Given a directed edge e = (u, v), we shall refer to user v as a follower of user u, while user u
is a followee of user v. User v ∈V can forward information I from user u ∈V. User u decides
whether to spread some information or not based on: (i) the information type I ∈{0, 1};
(ii) the features xu of user u; and (iii) the set of its neighbors Nu ≜{v ∈V : (u, v) ∈E},
who forwarded the information earlier.
Edge types.
In this paper, we say that an event occurs when a user (follower) forwards
the information from one of its followees. This is represented by the edge over which this
event occurs, which in turn is represented by a pair of features (xu, xv) that correspond to
the end users; Our edge-based model views each edge e ∈E as a communication channel
associated with a given weight. Speciﬁcally, we classify the edges into Z ∈N types, so that
each edge e ∈E has an associated weight Wu,v ≜We ∈Z, where Z ≜{0, 1, . . . , Z −1}.
An edge with a larger weight is more likely to spread misinformation, i.e., an edge with
weight 0 is the type of edges that are more likely to spread genuine information, while an
edge with weight Z −1 is more likely to spread fake information. Accordingly, we assume
that Wu,v = f(xu, xv), where f : E →Z is an edge-classiﬁer function. Finally, we denote
by W the |E| × |E| matrix with [W]u,v = Wu,v, for any (u, v) ∈E. As will be explained
later on, our decision algorithm will be based on partial observations of these weights, i.e.,
retweets are represented by the weights. Fig. 1 demonstrates the strong correlation between
the information type I and the SVM classiﬁcation score of the edge-based model on the
Weibo dataset [6]. More speciﬁcally, Fig. 1a illustrates that the scores of edges involved in
spreading genuine information are concentrated around zero, while Fig. 1b shows that the
scores of edges involved in spreading fake information are concentrated around one. This
is consistent with the fact that an edge with a higher weight is more likely to spread fake
information.
Probabilistic information ﬂow.
Given the above setup for the underlying social net-
work, and edge types, we next deﬁne the way information ﬂow/spread over the network. We
focus on a single information source s ∈V, and assume the following probabilistic model. Let
P denote the set of all possible directed paths in G starting at s. The information I can ﬂow
4

## Page 5

1
2
5
12
13
14
6
15
3
7
16
17
18
8
19
20
9
21
22
4
10
23
24
11
25
26
27
WP1
2,1
WP1
6,2
WP1
14,6
WP2
3,1
WP2
7,3
WP2
19,7
WP3
23,1
WP4
4,1
WP4
24,4
Figure 2: A partial social media graph with a single information source at s = 1. Each
weighted path in the graph corresponds to a diﬀerent Markov chain.
over one or more of these paths. As mentioned above, an event occurs when a user retweets
the information. Accordingly, for each possible path in G we model the sequence of consecu-
tive retweets (or, equivalently edges) as a ﬁrst-order Markov chain. Mathematically, for any
path P ∈P in G, let its vertex sequence be denoted by P = (vP
1 , vP
2 , . . . , vP
|P|). Then, each such
path is associated with a ﬁrst-order homogeneous Markov chain WP ≜{WvP
i ,vP
i+1}|P|−1
i=1
with Z
states. In particular, each retweet of I by a followee-follower pair (u, v) ∈P is represented by
the weight (edge-type) WP
u,v, which in turn corresponds to a feature pair (xu, xv), that is clas-
siﬁed into one of Z states by f(·), as described above. Given I, we let αI(z|z′), for z, z′ ∈Z,
denote the edge transition probabilities that depend on the underlying information type.
These transition probabilities clearly depend on whether the underlying information is gen-
uine or not. Indeed, it is reasonable that α0(0|0) > α1(0|0), namely, the transition between
two edges of type “0” is more probable if a genuine message is being propagated. The mo-
tivation is that if the transition probabilities α0(·|·) and α1(·|·) are “suﬃciently distinctive”,
we should be able decide whether genuine or fake information is propagated. Furthermore,
for any u ∈Ns we deﬁne the initial probabilities ηI(z) ≜P(Ws,u = z|I), for any z ∈Z, and
I ∈{0, 1}. Later on, we will devise a data-based oﬄine algorithm to learn all the above
terms. Tables 1 and 2 show the empirical transition probabilities α0(·|·) and α1(·|·), and
the initial probabilities η(·), estimated using the Weibo dataset, with Z = 4, respectively.
Throughout this paper, we follow the experimental setting in Section 4. Thenceforth, with
some abuse of notation, for a given path P ∈P, we let WP
v→u denote the trajectory of edge
weights starting from user v ∈V and ending at user u ∈V. Fig. 2 gives an example of a
social media graph with multiple Markov chains paths.
Observations and learning problem.
We now formulate the misinformation detection
problem as a sequential hypothesis testing problem. Firstly, we deﬁne the type of observa-
tions available for testing. When complete network and diﬀusion information are known,
the information spreading trace forms a tree with its paths forming Markov chains. Alas, in
practice, it is often not the case due to missing information and partial observations [3, 4],
thus, we assume that we observe only arbitrary parts of the information spreading traces
which we denote by the sequence {Zℓ}ℓ≥1, where Zℓ= WP
u,v, for some path P ∈P, and edge
(u, v) ∈E. Given I, the sequence {Zℓ}ℓ≥1 is subjected to a joint probability law that is
5

## Page 6

0
1
2
3
0
0.159
0.029
0.191
0.621
1
0.959
0.001
0.001
0.039
2
0.057
0.017
0.016
0.91
3
0.057
0.145
0.027
0.771
(a) Real news
0
1
2
3
0
0.659
0.017
0.028
0.297
1
0.065
0.015
0.021
0.899
2
0.064
0.011
0.075
0.85
3
0.026
0.004
0.006
0.964
(b) False news
Table 1: Edge transition probability matrices α(·|·) from the Weibo dataset, for Z = 4.
0
1
2
3
Real News
0.872
0.004
0.003
0.120
False News
0.101
0.006
0.015
0.876
Table 2: Edge initial probabilities η(·) from the Weibo dataset, for Z = 4.
1
2
5
12
13
14
6
15
3
7
16
17
18
8
19
20
9
21
22
4
10
23
24
11
25
26
27
Z1
Z2
Z3
Z4
Z5
Z6
Figure 3: An illustration of a possible sequence of observations {Z1, . . . , Z6} in a social media
graph with a single source s = 1.
governed by the transition probabilities αI(·|·). Below, when the underlying information is
genuine (fake) I = 0 (I = 1), we say that {Zℓ}ℓ≥1 is generated from L0 (L1). An illustration
of this observation model is given in Fig. 3. Compared to Fig. 2, it can be seen that the
actual observations are subsamples of some edges on the complete paths.
Our learning problem is formulated as follows. Consider a sequence {Zℓ}ℓ≥1 that obey
one of the two hypotheses, and the audit is tasked with distinguishing between the two
hypotheses,
H0 : I = 0
vs.
H1 : I = 1,
(1)
namely, the underlying information is genuine (null hypothesis) or fake (alternative hypothe-
sis). In terms of the prior distribution, we assume that hypothesis H1 occurs with probability
π1 = π, and H0 occurs with probability π0 ≜1 −π, for some π ∈[0, 1]. The audit is tasked
with distinguishing between the two hypotheses in a way that minimizes a combination of the
error probability and the propagation cost, as we deﬁne in the sequel. Consider a probability
space (Ω, F, Pπ), where Pπ is the probability measure deﬁned as follows
Pπ = (1 −π) P0 +π P1,
(2)
6

## Page 7

with P0 and P1 being the probability measures under the null and alternative hypotheses,
respectively, such that, under PI, the sequence {Zℓ}ℓ≥1 is generated from LI, for I = {0, 1}.
Now, note that the measures P0 and P1 are mutually singular, since the likelihood ratio
yields
Λℓ≜P1(Z1, . . . , Zℓ)
P0(Z1, . . . , Zℓ)
ℓ→∞
−−−→
(
0,
a.s. under P0
∞,
a.s. under P1 .
(3)
That is, we can tell the distributions apart from the limiting value of the likelihood ratio.
So, if we observe {Zℓ}ℓ≥1 we can decide perfectly between the two hypotheses. In this paper,
however, there is an additional cost for sampling. Speciﬁcally, suppose we observe {Zℓ}ℓ≥1
sequentially, generating the natural ﬁltration {Fℓ}ℓ≥1, with
Fℓ≜{Z1, Z2, . . . , Zℓ},
(4)
and F0 ≜(Ω, ∅). Clearly, a tradeoﬀdevelops between the decision accuracy and the potential
damage of spreading misinformation. The arising optimization problem can be examined in
the context of sequential decision rule. Let T ∈T be the random stopping time at which the
type of information is declared, where T denotes the set of all stopping times T ≥1 with
respect to the ﬁltration {Fℓ}, and a sequence {δℓ} of terminal decision rules, where δℓis an
Fℓ-measure function taking the values {0, 1}. Let D denote the set of all such δ. Then, a
sequential decision rule is deﬁned as
δT ≜
∞
X
ℓ=0
δℓ1{T=ℓ},
(5)
where 1{T=ℓ} is the indicator function that gets 1 when ℓis the stopping time T and 0
otherwise. Obviously, δT is an FT-measure as well and also taking the values {0, 1}. Now, a
sequential decision rule is described as (T, δ), in which T declares the time to stop sampling,
and once T is given, δT takes the values 0 or 1 declaring which hypotheses to accept: H0
for genuine information and H1 for fake information, correspondingly. For a given (T, δ), we
deﬁne the average cost of errors due to misdetection, as
ce(T, δT) ≜cI PH0(δT = 1) + cII PH1(δT = 0),
(6)
where cI, and cII are the costs of type-I error (news is declared as misinformation) and
type-II error (misinformation is declared as news). Furthermore, the propagation cost due
to spreading misinformation is given by c E[T1H1], where c ∈R+ is the cost of spreading
misinformation at each time slot, and 1H1 is the indicator function that gets the value 1
when hypothesis H1 is true and 0 otherwise. Our main goal is to ﬁnd the stopping time T
and decision rule δ that minimize the combination of the error probability and propagation
costs, i.e.,
inf
T∈T ,δ∈D ce(T, δT) + c E [T1H1] .
(7)
In the following section, we derive closed-form expressions for the optimal detector and
stopping rule, along with several performance guarantees. We then propose model and data
driven algorithms, which in addition to the above, classify the edge types and estimate the
transition probabilities.
7

## Page 8

vP
Iℓ(P)
uP
Iℓ(P)
vℓ
uℓ
ZP
Iℓ(P) = WP
uP
Iℓ(P),vP
Iℓ(P)
Zℓ= WP
uℓ,vℓ
Figure 4: An illustration of a path P ∈Pℓwhich forms the Markov-chain between ZP
Iℓ(P) =
WP
uP
Iℓ(P),vP
Iℓ(P) and Zℓ= WP
uℓ,vℓ.
3
Main Results
3.1
Optimal sequential test
In this subsection, we ﬁnd the optimal test minimizing the objective function in (7). To that
end, we start by presenting a few deﬁnitions. Let
Πℓ≜P (H1|Fℓ) ,
(8)
denote the posterior probability. Also, let AI(Zℓ|Fℓ−1) denote the conditional probability of
an observation Zℓgiven all the previous observations and the hypothesis HI, namely,
AI(Zℓ|Fℓ−1) ≜P (Zℓ|Fℓ−1, HI) .
(9)
Let Pℓdenote the set of all directed paths leading to Zℓfrom the source s. For each path
P ∈Pℓ, we let ZP
Iℓ(P) denote the last observation in the sequence (Z1, . . . , Zℓ−1) before Zℓ
in the path. Then, note that the sequence of observations (ZP
Iℓ(P), WP
Iℓ(P)+1→ℓ−1, ZP
ℓ) across
the path P form a ﬁrst-order Markov chain. This is illustrated in Fig. 4. For simplicity of
notation, we deﬁne the path score probability measure as
µI(P) ≜P (P|Fℓ−1, HI) ,
(10)
for any P ∈Pℓ, and I ∈{0, 1}. We show in the proof of Theorem 1 below that µI(P) is given
by (11), shown at the top of the next page, where J P
ℓ−1 is a set of indices of the observations
in the sequence (Z1,. . ., Zℓ−1) that belong to the path P. We are now in a position to state
our ﬁrst main result.
Theorem 1 (Posterior probability recursion). The following recursive relation holds,
Πℓ+1 =
ΠℓA1(Zℓ+1|Fℓ)
ΠℓA1(Zℓ+1|Fℓ) + (1 −Πℓ)A0(Zℓ+1|Fℓ),
(12)
8

## Page 9

µI(P) =
Q
i∈J P
ℓ−1
P
zi−1
Ii(P)+1∈Zi−Ii(P)−1
iQ
j=Ii(P)+1
αI(zj|zj−1)
P
P′∈Pℓ
Q
i∈J P
ℓ−1
P
zi−1
Ii(P′)+1∈Zi−Ii(P′)−1
iQ
j=Ii(P′)+1
αI(zj|zj−1)
(11)
AI(Zℓ|Fℓ−1) = EµI


X
zi−1
Ii(P)+1∈Zi−Ii(P)−1
iY
j=Ii(P)+1
αI(zj|zj−1)


(13)
where AI(Zℓ|Fℓ−1) = ηI(Zℓ) if Zℓis connected directly to the source, otherwise, we have (13),
shown at the top of the next page.
Next, we show that the optimization problem in (7) can cast as the following optimal
stopping problem.
Theorem 2 (Optimal sequential test). The optimal stopping problem in (7) is equivalent to
the following optimal stopping problem,
inf
T,δ ce(T, δT) + c E[T1H1] = inf
T E[g(ΠT) + cTΠT],
(14)
with g(π) ≜min{cIIπ, cI(1 −π)}. Furthermore, the optimal test minimizing (7) is given by
δT = 1{cIIΠT>cI(1−ΠT)}.
Theorem 2 implies that the only variable in the equivalent optimal stopping problem
is the stopping time T.
Furthermore, it can be seen that we converted the problem of
minimizing over both the stopping time and the decision rules to the new problem in (14),
where we only need to ﬁnd the optimal stopping time. Accordingly, we can ﬁnd the optimal
stopping policy in two steps: ﬁrst ﬁnd the optimal stopping time T by solving (14), and then
ﬁnd the optimal decision rule δT. We next characterize the optimal stopping time rule. To
that end, we need a few deﬁnitions. Let Tℓdenote the subset of the set of all stopping times
T with respect to the ﬁltration Fℓsatisfying P(T ≥ℓ) = 1, for all ℓ≥1. For ℓ= 1, 2, . . . we
deﬁne the sequence {sℓ}ℓ≥1,
sℓ(π, zℓ
1) ≜inf
T∈TℓE[g(ΠT) + cTΠT|Πℓ= π, Fℓ= zℓ
1],
(15)
where s0(π, ∅) ≜g(π). Note that sℓ(π, zℓ
1) is the minimum expected total cost if the algorithm
is obligated to stop at time T ≥ℓ, conditioned on the information up to ℓ. Also, deﬁne
¯sℓ
 π, zℓ
1

= sℓ
 π, zℓ
1

−cℓπ, for ℓ≥1. for ℓ≥1. We have the following result.
9

## Page 10

Theorem 3 (Optimal stopping time). The optimal stopping time T⋆, achieving the minimum
in (14), is given by,
T⋆= inf{ℓ∈N: Πℓ/∈(πlow(zℓ
1), πup(zℓ
1))},
(16)
and the optimal decision rule is
δT⋆(zT⋆
1 ) =
(
0,
ΠT⋆≤πlow(zT⋆
1 )
1,
ΠT⋆> πup(zT⋆
1 ),
(17)
with
πlow(zℓ
1) = sup

0 ≤π ≤
cI
cI + cII
: ¯sℓ
 π, zℓ
1

= cIIπ

,
(18)
πup(zℓ
1) = inf

cI
cI + cII
≤π ≤1 : ¯sℓ
 π, zℓ
1

= cI(1 −π)

.
(19)
Furthermore, we have,
¯sℓ(π, zℓ
1) = min

g(π) , cπ
+ E

¯sℓ+1(Πℓ+1, Fℓ+1)|Πℓ= π, Fℓ= zℓ
1
	
.
(20)
The equivalence of our problem to the well-known optimal stopping problem in Theorem
2 allows us to achieve an optimal quickest solution. Theorem 3 states that our solution is
represented by time dependent pairs of lower and upper thresholds (πlow, πup), where each
pair corresponds to a sequence of the observed edges zℓ
1. The posterior probability thresholds
in Theorem 3 are calculated in real-time and are updated at every iteration ℓaccording to
the solution of the Bellman equation (20). The stopping rule in (16) keeps track of the
posterior probability Πℓ. Once this posterior goes out of the threshold range, the auditing
stops and makes a decision according to (17).
3.2
Statistical guarantees
In this subsection we provide a few statistical guarantees associated with the optimal decision
rule we derived in the previous subsection. We start by the following observation where we
represent the optimal decision rule as a sequential probability ratio test (SPRT), as SPRT’s
are known to exhibit minimal expected stopping time among all sequential decision rules
having given error probabilities.
Theorem 4 (SPRT representation). Consider the optimization problem in (7). If πlow(zℓ
1) <
π < πup(zℓ
1), for all ℓ= 1, 2, . . ., then the optimal solution given in Theorem 3 can be
equivalently deﬁned as an SPRT with boundaries Blow and Bup, namely,
T⋆= inf

ℓ∈N : Λℓ/∈(Blow(zℓ
1), Bup(zℓ
1))
	
,
(21)
10

## Page 11

where
Λℓ≜
ℓY
i=1
P (Zi|Fi−1, H1)
P (Zi|Fi−1, H0) = A1(Zℓ|Fℓ−1)
A0(Zℓ|Fℓ−1)Λℓ−1,
(22)
for ℓ= 1, 2, . . ., with Λ0 ≜1. The thresholds Blow and Bup are given by
Blow(zℓ
1) = 1 −π
π
·
πlow(zℓ
1)
1 −πlow(zℓ
1),
(23)
Bup(zℓ
1) = 1 −π
π
·
πup(zℓ
1)
1 −πup(zℓ
1).
(24)
The decision rule is given by,
δT⋆(zT⋆
1 ) =
(
0,
ΛT⋆≤Blow(zT⋆
1 )
1,
ΛT⋆> Bup(zT⋆
1 ).
(25)
Proposition 1. For each ℓ, ﬁx 0 < Blow ≤1 ≤Bup < ∞. Let,
Pe,1 ≜PH0 (δT⋆= 1) ,
(26)
Pe,2 ≜PH1 (δT⋆= 0) .
(27)
Then, the following relationship among the thresholds and the error probabilities hold,
Blow ≥
Pe,2
1 −Pe,1
,
(28)
Bup ≤1 −Pe,2
Pe,1
.
(29)
An important consequence of the above proposition is that the boundaries Blow and Bup
can be chosen to yield a given level of error probability performance. For example, if we wish
to design a test ϕ with approximate error probabilities pϕ and qϕ, then we can use Wald’s
approximations to choose boundaries Blow =
qϕ
1−pϕ and Bup = 1−qϕ
pϕ . Then, inequalities (28)
and (29) imply that the actual error probabilities are bounded according bu
Pe,1 ≤
pϕ
1 −qϕ
= pϕ(1 + O(qϕ)),
(30)
Pe,2 ≤
qϕ
1 −pϕ
= qϕ(1 + O(pϕ)).
(31)
Thus, for a small desired error probabilities, the actual error probabilities obtained by using
Wald’s approximations can be bounded by values that are quite close to their desired values.
And, in fact, these bounds will be tight in the limit of small error probabilities. Namely, we
can design a test that achieves with good accuracy error probabilities as small as desired.
Clearly, as the error probabilities get smaller, the range (Blow, Bup) gets larger so that the
stopping time would increase. This is consistent with (3), which shows that the hypothesis
can be accurately distinguished by the limit value of the likelihood ratio.
11

## Page 12

3.3
Model and data driven algorithms
In this subsection, we present our model and data driven algorithms. The pseudo-codes
of our misinformation detection procedure are given in Algorithms 1 and 2. Speciﬁcally,
Algorithm 1 is an oﬄine procedure that trains an edge-classiﬁer f(·), learns the transition
probabilities αI(·|·), and the initial probabilities ηI(·), for I = {0, 1}. The required dataset
contains both the social media graph and genuine/fake labeled information spreading traces
{tk}N
k=1, that include for each user the followee from whom the information was retweeted.
The online Algorithm 2 is an implementation of our proposed sequential detection procedure.
We next discuss the oﬄine and online routines in more detail.
3.3.1
Oﬄine Algorithm
Our oﬄine algorithm requires training dataset that contains both the social media graph
G and N labeled information spreading traces, which we denote by {tk}N
k=1. The spreading
traces are labeled as genuine or fake news and must include for each user the followee
from whom the information was retweeted. The social media graph data includes all the
connections between the users involved in the information spreading traces with each user
has an associated feature vector. In order to ﬁnd the function f(), that classiﬁes an edge
e = (u, v) to one of Z classes, we ﬁrst train a linear SVM that classiﬁes the information to
genuine or fake news. We calculate the average feature vector of each information trace, and
along with its corresponding label we generate the input for our SVM. The SVM returns a
value in [0, 1]. This range is divided into Z equal intervals that are assigned with an edge
type in a sequential manner. For example, if Z = 4, then we use the mapping [0, 0.25] →0,
(0.25, 0.5] →1, (0.5, 0.75] →2, and (0.75, 1] →3. Using this mapping we then estimate two
important terms:
• Initial probability: the probability of an edge of type z ∈Z to forwarded information
of type I = {0, 1} directly from the source s, using
ˆηI(z) =
PN
k=1
P|tk|−1
ℓ=1
1{e
tk
ℓ∈Estk ,W
etk
ℓ
=z}1{lk=I}
PN
k=1 |Estk|1{lk=I}
,
(32)
where etk
ℓ= (utk
ℓ, vtk
ℓ), and Estk is the set of edges that are connected to the source stk
in tk.
• Transition probabilities: we estimate
ˆαI(z|z′) =
PN
k=1
P|tk|−1
ℓ=1
1{W
etk
ℓ
=z,We′
ℓ
tk =z′}1{lk=I}
PN
k=1
P|tk|−1
ℓ=1
1{W
e′tk
ℓ
=z′}1{lk=I}
,
(33)
where e′tk
ℓis the adjacent edge of etk
ℓthat previously forwarded the information in the
trace tk; if no such edge exist, and etk
ℓis connected directly to the source stk, then the
indicator is nulliﬁed.
12

## Page 13

3.3.2
Online Algorithm
The inputs to our online procedure are: the social media graph G, partial information trace
{Zℓ}ℓ≥1 including the source s, edge classiﬁer f(·), initial probabilities ˆηI(·), and transition
probabilities ˆαI(·|·). The algorithm initializes the prior distribution of hypothesis H1 accord-
ing to the data, Π0 = π0. When an event Zℓoccurs, the algorithm calculates the conditional
transition probability AI using (13), and then updates Π1 using (12). We note that the
exact calculation of (20), and therefore the thresholds πlow, πup, entails taking expectation
over the entire collection of unobserved edges in E. Therefore, our online algorithm performs
a ﬁrst-order approximation and stops with the ﬁrst sign of convergence of Πℓ, that is, it
stops when |Πℓ+1 −Πℓ| < ǫ, for some initialized ǫ > 0. The information is declared to be
fake news if Π1 ≥π1, and genuine information, otherwise. Clearly, this approximation can
only impair the performance of our algorithm.
Algorithm 1 Misinformation training (Oﬄine)
Input: Social media graph, N information propagation traces {t1, . . . , tN} with labels L =
{l1, . . . , lN}, lk ∈{0, 1}. Each tk is a sequence of |tk| users vtk
ℓand feature vectors xv
tk
ℓ.
Output: f(·), ˆαI(·|·), ˆηI(·).
• For each user vℓin each information trace tk, obtain the followee-follower feature vector
(xuℓ, xvℓ) and compute
¯F(tk) =
1
|tk|−1
|tk|
P
ℓ=2
xu
tk
ℓ,
|tk|
P
ℓ=2
xu
tk
ℓ

.
• Train edge classiﬁer f(·) using SVM with input (¯F, L).
• Classify each edge e = (u, v) ∈E to We ←f(xu, xv).
• For all z ∈Z calculate ˆηI(z) using (32).
• For all z, z′ ∈Z calculate ˆαI(z|z′) using (33).
Algorithm 2 Misinformation Detection (Online)
Input: Social media graph, partial trace {Z1, Z2 . . .} with a known source s, f(·), ˆαI(·|·),
ˆηI(·).
Output: Genuine/Fake.
Initialize: ǫ, cI, cII, c, Π0 ←π1, Π1 ←π1 + 2ǫ, F0 ←∅
while |Π0 −Π1| ≥ǫ do
Π0 ←Π1, ℓ←ℓ+ 1, Fℓ←{Fℓ−1, Zℓ}
Calculate AI(Zℓ|Fℓ−1) using (13).
Update Π1 using (12).
if Π1 ≥π1 then return Fake
else return Genuine
4
Experiments
In our research we make use of the Weibo dataset [6]. Sina Weibo is China’s leading micro-
blogging service provider with eight times more users than Twitter. The dataset includes
13

## Page 14

Number of users
2,746,818
Number of Tweets
3,805,656
Number of events
4,664
Number of rumors
2,313
Number of non-rumors
2,351
Average time length/event
2,460.7 Hours
Average number of posts/event
816
Maximum number of posts/event
59,318
Minimum number of posts/event
10
Table 3: Details of the Weibo dataset
4,664 labeled information traces provided by Sina’s community management center1 with an
average of 816 retweets per trace. User features such as the number of followees, the number
of followers, the registration days, etc., were originally extracted from Sina Weibo API.2
Table 3 summarizes the details of the Weibo dataset. The social media graph structure G
is reconstructed by a union of all the information traces. Moreover, recall that our online
algorithm requires only partial information propagation traces in order to make a quick and
accurate decision. Since the Wiebo dataset contains the entire information propagation trace,
we have uniformly drawn 50% of its observations.
In our simulations, we divide the complete dataset into 80% training and 20% testing
(with 10% genuine news traces and 10% fake news traces). In addition, we take cI = cII = 10,
c = 0.05, ǫ = 0.001, and Z = 4. For simplicity of computation, in the calculation of AI,
we truncated the various paths in the graph to a predeﬁned maximal length; note that this
can only impair the performance of our algorithm. We compare our algorithm to QuickStop
[17], and the following state-of-the-art algorithms:
• SVM-TSu and SVM-TSa [7] are dynamic series-time structure (DSTS) based SVM
methods. The former takes in consideration the user features alone, while the later is
a fully conﬁgured model that utilizes all content-based, user-based and diﬀusion-based
features.
• DTCu and DTCa [1] are automatic methods for assessing the credibility of a given set
of tweets on Twitter based on decision trees. The ﬁrst mentioned method uses the user
features only, while the second method also uses the content-based features.
• SVM-RBFu and SVM-RBFa [18] are SVM-based detection methods with RBF kernel
function. The former uses only user features, whereas the later uses both user and
content-based features.
• CSI [13] is a hybrid deep model which is composed of three modules: (1) an RNN
to capture the temporal pattern of user activity, (2) fully connected layer for source
1https://service.account.Weibo.com
24http://open.weibo.com/wiki/API
14

## Page 15

0
2
4
6
8
10
12
14
16
18
0
0.2
0.4
0.6
0.8
1
Number of events
π1
Real news
Fake news
Figure 5: Examples of Πℓand T on real/fake news under our method.
characteristic learning based on the users behavior, and (3) integration module for
classiﬁcation.
• PPC-R, PPC-C and PPC-R+C [5] are detection models through propagation path
classiﬁcation for time-series data with a gated recurrent unit (GRU), a CNN, and a
combination of RNN and CNN, respectively.
For the purpose of comparison, we use the following performance metrics:
• Accuracy: the fraction of traces that are correctly classiﬁed.
• False positive (FP): the fraction of genuine news classiﬁed as fake news.
• False negative (FN): the fraction of fake news classiﬁed as genuine news.
• Detection time: the average number of events required to declare the type of detection).
Figures 5 and 6 present the evolution of Πℓon two traces, genuine and fake news, chosen
from the Weibo dataset, using our method, and as compared to QuickStop, respectively. Both
examples evidently show that our algorithm succeeds and declares the correct information
type, while QuickStop fails to do so, as it yields FN (see, Fig. 6a) and FP (see, Fig. 6b).
As expected, it is evident that our algorithm makes a quicker decision when a fake news
propagates, due to the propagation cost of spreading misinformation in (7). Speciﬁcally,
based on our experiments, our detection algorithm requires an average of 5.6 events for
misinformation detection and an average of 7.2 events for news detection; altogether, a
decision is made after 6.29 events on average. Keeping in mind that in our simulations, we
implemented a ﬁrst-order approximation of the exact algorithm in Theorem 3, it is reasonable
that the likelihood Πℓmay exceed the interval (πlow(zℓ
1), πup(zℓ
1)), before the convergence of
Πℓoccurs. In this case, our exact algorithm will arrive at quicker decisions with the same
accuracy rate.
15

## Page 16

πup
0
5
10
15
20
25
0
0.5
1
Iteration number
π1
Our method
QuickStop
(a) Real news
πlow
0
2
4
6
8
0.5
1
Iteration number
π1
Our method
QuickStop
(b) False news
Figure 6: Examples of Πℓand stopping time T under our method and QuickStop.
0
100
200
300
400
500
0.5
0.6
0.7
0.8
0.9
Number of Events
Accuracy
SVM-TSu
SVM-TSa
DTCu
DTCa
SVM-RBFu
SVM-RBFa
CSI
PPC-R
PPC-C
PPC-R+C
Quickstop
Our method
Figure 7: Detection accuracy as a function of ℓ.
Method
Accuracy
FP
FN
Decision Deadline
Quickstop
0.85
0.08
0.20
12.75
Ours
0.86
0.08
0.18
6.29
Table 4: Detailed comparison of our method and QuickStop.
Finally, Fig. 7 compares the accuracy of our algorithm to the previously mentioned
algorithms.
It is clear that our algorithm outperforms all of these algorithms, both in
terms of accuracy and detection time. Speciﬁcally, in Table 4 we zoom-in and compare our
algorithm to QuickStop. It can be seen that on average, our algorithm achieves the same
detection accuracy (as well as false positive (FP) and false negative (FN) rates), but roughly
in half the time.
16

## Page 17

5
Proofs
5.1
Proof of Theorem 1
Recall that Πℓ= P (H1|Fℓ), and note that Πℓis a Doob’s martingale with respect to the
ﬁltration Fℓ. Indeed,
E [Πℓ|Fℓ−1] = E [E [1H1|Fℓ] |Fℓ−1] = E [1H1|Fℓ−1] = Πℓ−1,
(34)
where we used the fact that Fℓ−1 ⊂Fℓ. According to the Bayes rule, we have
Πℓ= P (H1|Fℓ)
(35)
= P (Fℓ|H1) P (H1)
P (Fℓ)
(36)
=
π1 P (Fℓ|H1)
π1 P (Fℓ|H1) + π0 P (Fℓ|H0),
(37)
while the above joint probability distributions can be expressed as,
P (Fℓ|HI) = P (Z1, Z2, . . . , Zℓ|HI)
(38)
=
ℓY
i=1
AI(Zi|Fi−1).
(39)
Thus, we can write
1 −Πℓ
Πℓ
=
(1 −π1)
ℓQ
i=1
A0(Zi|Fi−1)
π1
ℓQ
i=1
A1(Zi|Fi−1)
,
(40)
which implies that
1 −Πℓ+1
Πℓ+1
= 1 −Πℓ
Πℓ
A0(Zℓ+1|Fℓ)
A1(Zℓ+1|Fℓ).
(41)
Therefore, we arrive at the following recursive relation,
Πℓ+1 =
ΠℓA1(Zℓ+1|Fℓ)
ΠℓA1(Zℓ+1|Fℓ) + (1 −Πℓ)A0(Zℓ+1|Fℓ).
(42)
It is left to derive an explicit formula for AI(Zi|Fi−1). Recall that Pℓis the set of all the
possible directed paths starting from the source s ∈V and ending at the current observation
Zℓ. Then, using the law of total probability, AI(Zℓ|Fℓ−1) can be explicitly rewritten as,
AI(Zℓ|Fℓ−1) = P (Zℓ|Fℓ−1, HI)
(43)
17

## Page 18

=
X
P∈Pℓ
P (P|Fℓ−1, HI) P (Zℓ|Fℓ−1, P, HI) .
(44)
We next ﬁnd formulas for the probabilities inside the summation in (44), starting with
P (P|Fℓ−1, HI). There are two cases to consider here. If Zℓis connected to the source s
directly, then we clearly have,
AI(Zℓ|Fℓ−1) = ηI(Zℓ).
(45)
Otherwise, let us denote the set of observations in the sequence (Z1, . . . , Zℓ−1) that belong
to the path P by F P
ℓ−1 and their indices by J P
ℓ−1. Then the independency of the observations
along with Bayes theorem and the law of total probability imply that,
P (P|Fℓ−1, HI) = P
 P|F P
ℓ−1, HI

(46)
=
Q
i∈J P
ℓ−1 P (Zi|Fi−1, P, HI)
P
P′∈Pℓ
Q
i∈J P
ℓ−1 P (Zi|Fi−1, P′, HI),
(47)
where we have used the fact that P(P|HI) = |Pℓ|−1. Thus, we see that ﬁnding a formula
for (44) boils down to ﬁnding a formula for P (Zi|Fi−1, P, HI), for any 1 ≤i ≤ℓ, and P ∈Pℓ.
By the Markov property, the conditional probability of Zi given the path P and the sequence
of observations Fi−1 depends only on the last observation in the sequence (Z1, . . . , Zi−1) in
the path P. Recall that we denote the index of this last observation by Ii(P). Also, recall
that the sequence of observations (ZP
Ii(P), WP
Ii(P)+1→i−1, ZP
i ) across the path P form a ﬁrst-order
Markov chain. Otherwise, by the law of total probability, we have
P (Zi|Fi−1, P, HI)
=
X
zi−1
Ii(P)+1∈Zi−Ii(P)−1
iY
j=Ii(P)+1
αI(zj|zj−1),
(48)
where zi−1
Ii(P)+1 = (zIi(P)+1, . . . , zi−1). Thus, we get (49), shown at the top of the next page.
P (P|Fℓ−1, HI) =
Q
i∈J P
ℓ−1
P
zi−1
Ii(P)+1∈Zi−Ii(P)−1
Qi
j=Ii(P)+1 αI(zj|zj−1)
P
P′∈Pℓ
Q
i∈J P
ℓ−1
P
zi−1
Ii(P′)+1∈Zi−Ii(P′)−1
Qi
j=Ii(P′)+1 αI(zj|zj−1)
.
(49)
Substituting (45), (48), and (49) in (44), we readily obtain an expression for AI(Zℓ|Fℓ−1).
Speciﬁcally, recall the deﬁnition of the measure µI(P) in (10). Then,
AI(Zℓ|Fℓ−1)
= EµI


X
zi−1
Ii(P)+1∈Zi−Ii(P)−1
iY
j=Ii(P)+1
αI(zj|zj−1)

.
(50)
18

## Page 19

5.2
Proof of Theorem 2
We separate this proof into two parts, starting with the propagation cost. We ﬁrst show
that E [T1H1] = E [TΠT]. According to the law of total expectation we have
E [T1H1] = E [E [T1H1|T]]
(51)
= E [T E [1H1|T]]
(52)
= E [T E [E [1H1|FT, T]]]
(53)
= E [T E [E [1H1|FT] |T]]
(54)
= E [T E [P (H1|FT) |T]]
(55)
= E [T E [ΠT|T]]
(56)
= E [TΠT] .
(57)
Next, we analyze the average costs of errors. Speciﬁcally, we will show that ce(T, δT) =
E [min{cIIΠT, cI(1 −ΠT)}], for the likelihood ratio test δT. Indeed,
ce(T, δT) = cI PH0 (δT = 1) + cII PH1 (δT = 0)
(58)
=
∞
X
ℓ=1
cI P(δℓ= 1, H0|T = ℓ) P (T = ℓ)
+ cII P(δℓ= 0, H1|T = ℓ) P (T = ℓ)
(59)
=
∞
X
ℓ=1
cI E

1{δℓ=1}1H0|T = ℓ

P (T = ℓ)
+ cII E

1{δℓ=0}1H1|T = ℓ

P (T = ℓ)
(60)
=
∞
X
ℓ=1
(cI E

E

1{δℓ=1}1H0|Fℓ

|T = ℓ

+ cII E

E

1{δℓ=0}1H1|Fℓ

|T = ℓ

) P (T = ℓ)
(61)
=
∞
X
ℓ=1
E

cI1{δℓ(Fℓ)=1}(1 −Πℓ) + cII1{δℓ(Fℓ)=0}Πℓ|T = ℓ

· P (T = ℓ)
(62)
≥
∞
X
ℓ=1
E [min {cI(1 −Πℓ), cIIΠℓ}|T = ℓ] P (T = ℓ)
(63)
= E [min {cI(1 −Πℓ), cIIΠℓ}] .
(64)
Finally, we note that (63) holds with equality for the likelihood-ratio test rule, i.e., δT =
1{cIIΠT>cI(1−ΠT)}, which concludes the proof.
5.3
Proof of Theorem 3
Consider a two-dimensional stochastic process {Xℓ: ℓ= 0, 1, . . .} with state space S =
[0, 1] × Z. Let X0 = [π, m]T, and consider a family of measures

P(π,m) : [π, m]T ∈S
	
, such
19

## Page 20

that
P(π,m)
 X0 = [π, m]T
= 1.
(65)
We denote the expectation under P(π,m) by E(π,m). Note that the ﬁrst entry of Xℓevolves
in time ℓaccording to the recursion rule (12), and the second entry increases by ℓunits,
more speciﬁcally Xℓ= [Πℓ, m + ℓ]T. Therefore, the problem in (14) is a special case of the
following optimal stopping problem (with m = 0),
sup
T∈T1
E(π,m) h(XT),
(66)
where
h(x) ≜−g(π) −cmπ,
x = [x, π]T ∈S,
(67)
and recall that g(π) = min{cIIπ, cI(1 −π)}. Since supT∈T h(XT) ≤c(−m)+π, the general
inﬁnite horizon case of the optimal stopping theory [12] states that the stopping time that
solves problem (14) is given by
T⋆= inf{ℓ∈N : h(Xℓ) = γℓ(Xℓ, Fℓ)},
(68)
where for ℓ= 1, 2, . . .,
γℓ(x, zℓ
1) ≜sup
T∈Tℓ
E(π,m)|Fℓ

h(XT)|Fℓ= zℓ
1

.
(69)
Clearly, we have γℓ(x, zℓ
1) = −sℓ(π, zℓ
1) −cmπ, where
sℓ(π, zℓ
1) = −sup
T∈Tℓ
E(π,0)|Fℓ

h(XT)|Fℓ= zℓ
1

(70)
= inf
T∈TℓE

g(ΠT) + cTΠT|Πℓ= π, Fℓ= zℓ
1

.
(71)
Note that sℓ(π, zℓ
1) is the minimum expected total cost if the algorithm is obligated to stop
at time T ≥ℓ, conditioned on the information up to time ℓ. According to the general inﬁnite
horizon case of the optimal stopping theory the sequence {γℓ} satisﬁes the condition
γℓ(x, zℓ
1) = max {h(x),
E(π,m)|Fℓ

γℓ+1(Xℓ+1, Fℓ+1)|Fℓ= zℓ
1
	
.
(72)
Therefore,
sℓ(π, zℓ
1) = min {g(π) + cℓπ,
E

sℓ+1(Πℓ+1, Fℓ+1)|Πℓ= π, Fℓ= zℓ
1
	
.
(73)
The above implies (76),
¯sℓ
 π, zℓ
1

= sℓ
 π, zℓ
1

−cℓπ
(74)
= min

g(π) , E

sℓ+1(Πℓ+1, Fℓ+1) −c(ℓ+ 1)π|Πℓ= π, Fℓ= zℓ
1

+ cπ
	
(75)
20

## Page 21

= min

g(π) , E

¯sℓ+1(Πℓ+1, Fℓ+1)|Πℓ= π, Fℓ= zℓ
1

+ cπ
	
.
(76)
Thus, at optimal stopping time T⋆, we have
¯sℓ
 π, zT⋆
1

= g(π).
(77)
Furthermore, if ¯sℓ
 π, zℓ
1

= g(π) and cII < cI(1−π), then ¯sℓ
 π, zℓ
1

= cIIπ, and the information
is declared as genuine; otherwise, it is declared as fake. Hence,
πlow(zℓ
1) = sup

π ≤
cI
cI + cII
: ¯sℓ
 π, zℓ
1

= cIIπ

,
(78)
and
πup(zℓ
1) = inf

cI
cI + cII
≤π : ¯sℓ
 π, zℓ
1

= cI(1 −π)

.
(79)
5.4
Proof of Theorem 4
If πlow ≤π ≤πup, for all ℓ= 1, 2, . . ., then Blow and Bup, deﬁned in (23) and (24), respectively,
satisfy 0 < Blow ≤1 ≤Bup < ∞. From Theorem 3 the stopping time (16) may be written
T⋆= inf{ℓ∈N : Πℓ/∈(πlow(zℓ
1), πup(zℓ
1))},
(80)
but because,
Πℓ= P(H1|Fℓ)
(81)
=
π P(Fℓ|H1)
π P(Fℓ|H1) + (1 −π) P(Fℓ|H0)
(82)
=
πΛℓ
πΛℓ+ (1 −π),
(83)
we see that Πℓ/∈(πlow(zℓ
1), πup(zℓ
1)) is equivalent to Λℓ/∈(Blow(zℓ
1), Bup(zℓ
1)) with Blow(zℓ
1) and
Bup(zℓ
1), deﬁned in (23) and (24), respectively. Thus, the stopping time (21) is equivalent to
(16). Similarly, the decision rule (17) may be written as,
δT⋆=
(
0,
ΠT⋆≤πlow(zT⋆
1 )
1,
ΠT⋆> πup(zT⋆
1 ),
(84)
but again Πℓ≤πlow(zℓ
1) and Πℓ≥πup(zℓ
1) are equivalent to Λℓ≤Blow(zℓ
1) and Λℓ> Bup(zℓ
1),
respectively, for all ℓ≥1. Hence, the decision rule (25) is equivalent to (17).
5.5
Proof of Proposition 1
Recall that for each ℓ, we ﬁx 0 < Blow ≤1 ≤Bup < ∞. We let σT⋆= σ(z1, . . . , zT⋆) be the
σ-algebra. Consider the following chain of equations,
Pe,1 = PH0 (δT⋆= 1)
(85)
21

## Page 22

=
1
ZT⋆
X
FT⋆∈σT⋆
PH0 (δT⋆= 1|FT⋆)
(86)
=
1
ZT⋆
X
FT⋆∈σT⋆
PH0 (ΛT⋆≥Bup|FT⋆)
(87)
=
1
ZT⋆
X
FT⋆∈σT⋆
E

1{ΛT⋆≥Bup}1H0|FT⋆
(88)
=
1
ZT⋆
X
FT⋆∈σT⋆
∞
X
ℓ=1
E

1{ΛT⋆≥Bup}1H0|FT⋆, T⋆= ℓ

· P(T⋆= ℓ)
(89)
≤
1
ZT⋆
X
FT⋆∈σT⋆
∞
X
ℓ=1
1
Bup
E

ΛT⋆1{ΛT⋆≥Bup}1H0|FT⋆, T⋆= ℓ

· P(T⋆= ℓ)
(90)
=
1
ZT⋆
X
FT⋆∈σT⋆
∞
X
ℓ=1
1
Bup
E

1{ΛT⋆≥Bup}1H1|FT⋆, T⋆= ℓ

· P(T⋆= ℓ)
(91)
=
1
ZT⋆
X
FT⋆∈σT⋆
1
Bup
E

1{ΛT⋆≥Bup}1H1|FT⋆
(92)
= PH1(ΛT⋆≥Bup)
Bup
(93)
= 1 −Pe,2
Bup
,
(94)
where (87) holds due to the optimally of the stopping time which assures that T⋆is almost
surely ﬁnite, (90) holds since ΛT⋆≥Bup, and ﬁnally (91) holds due to the fact the all events
{T⋆= ℓ}, {Fℓ∈σT⋆} and {Λℓ≥Bup} are in Fℓ, and so,
E

Λℓ1{Λℓ≥Bup}1H0|FT⋆, T⋆= ℓ

=
Z
{Λℓ≥Bup,FT⋆∈σT⋆,T⋆=ℓ}
Λℓd P0
(95)
=
Z
{Λℓ≥Bup,FT⋆∈σT⋆T⋆=ℓ}
d P1
(96)
= E

1{Λℓ≥Bup}1H1|FT⋆, T⋆= ℓ

.
(97)
Finally, a similar argument gives Pe,2 ≤Blow(1 −Pe,1), which concludes the proof.
22

## Page 23

6
Conclusion
This paper introduces a quickest misinformation detection algorithm based on a realistic
probabilistic model of information propagation through a social media platform. The prob-
lem is formulated and solved as an optimal stopping problem that minimizes the combination
of the error probability and the stopping time. Our numerical results with a real-world data
demonstrate that our algorithm outperforms state-of-the-art early misinformation detection
algorithms. As an interesting direction for future research, while in this work, we considered
a hard decision problem between two possible hypotheses (genuine or fake), it is more rea-
sonable and robust to consider a softer decision problem, with multiple hypotheses, or even
a sequential estimation problem, where the parameter to be estimated reﬂects the level of
genuineness/fakeness.
References
[1] C. Castillo, M. Mendoza, and B. Poblete. Information credibility on twitter. In Pro-
ceedings of the 20th international conference on World wide web, pages 675–684, 2011.
[2] T. Chen, X. Li, H. Yin, and J. Zhang.
Call attention to rumors: Deep attention
based recurrent neural networks for early rumor detection. In Trends and Applications
in Knowledge Discovery and Data Mining: PAKDD 2018 Workshops, BDASC, BDM,
ML4Cyber, PAISI, DaMEMO, Melbourne, VIC, Australia, June 3, 2018, Revised Se-
lected Papers 22, pages 40–52. Springer, 2018.
[3] F. Jin, E. Dougherty, P. Saraf, Y. Cao, and N. Ramakrishnan. Epidemiological mod-
eling of news and rumors on twitter. In Proceedings of the 7th Workshop on Social
Network Mining and Analysis, SNAKDD ’13, New York, NY, USA, 2013. Association
for Computing Machinery.
[4] S. Kwon, M. Cha, K. Jung, W. Chen, and Y. Wang. Prominent features of rumor
propagation in online social media. In 2013 IEEE 13th International Conference on
Data Mining, pages 1103–1108, 2013.
[5] Y. Liu and Y.-F. Wu. Early detection of fake news on social media through propagation
path classiﬁcation with recurrent and convolutional networks.
In Proceedings of the
AAAI conference on artiﬁcial intelligence, volume 32, 2018.
[6] J. Ma, W. Gao, P. Mitra, S. Kwon, B. J. Jansen, K.-F. Wong, and M. Cha.
De-
tecting rumors from microblogs with recurrent neural networks. In Proceedings of the
Twenty-Fifth International Joint Conference on Artiﬁcial Intelligence, IJCAI’16, page
3818–3824. AAAI Press, 2016.
[7] J. Ma, W. Gao, Z. Wei, Y. Lu, and K.-F. Wong.
Detect rumors using time series
of social context information on microblogging websites.
In Proceedings of the 24th
ACM international on conference on information and knowledge management, pages
1751–1754, 2015.
23

## Page 24

[8] J. Ma, W. Gao, and K.-F. Wong. Detect rumors in microblog posts using propaga-
tion structure via kernel learning. In Proceedings of the 55th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), pages 708–717,
Vancouver, Canada, July 2017. Association for Computational Linguistics.
[9] J. Ma, W. Gao, and K.-F. Wong. Rumor detection on Twitter with tree-structured
recursive neural networks. In Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 1980–1989, Melbourne,
Australia, July 2018. Association for Computational Linguistics.
[10] F. Monti, F. Frasca, D. Eynard, D. Mannion, and M. M. Bronstein. Fake news detection
on social media using geometric deep learning. arXiv:1902.06673, 2019.
[11] J. Paschen. Investigating the emotional appeal of fake news using artiﬁcial intelligence
and human contributions. Journal of Product & Brand Management, 05 2019.
[12] H. V. Poor and O. Hadjiliadis. Quickest detection. Cambridge University Press, 2008.
[13] N. Ruchansky, S. Seo, and Y. Liu. Csi: A hybrid deep model for fake news detec-
tion. In Proceedings of the 2017 ACM on Conference on Information and Knowledge
Management, pages 797–806, 2017.
[14] E. Shearer and K. Matsa. News use across social media platforms 2018. Pew Research
Center, 2018.
[15] K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu. Fake news detection on social media:
A data mining perspective. SIGKDD Explor. Newsl., 19(1):22–36, sep 2017.
[16] S. Vosoughi, D. Roy, and S. Aral. The spread of true and false news online. Science,
359(6380):1146–1151, 2018.
[17] H. Wei, X. Kang, W. Wang, and L. Ying. Quickstop: A markov optimal stopping ap-
proach for quickest misinformation detection. Proceedings of the ACM on Measurement
and Analysis of Computing Systems, 3(2):1–25, 2019.
[18] F. Yang, Y. Liu, X. Yu, and M. Yang. Automatic detection of rumor on sina weibo.
In Proceedings of the ACM SIGKDD workshop on mining data semantics, pages 1–7,
2012.
[19] Z. Zhao, P. Resnick, and Q. Mei. Enquiring minds: Early detection of rumors in social
media from enquiry posts. In Proceedings of the 24th international conference on world
wide web, pages 1395–1405, 2015.
24
