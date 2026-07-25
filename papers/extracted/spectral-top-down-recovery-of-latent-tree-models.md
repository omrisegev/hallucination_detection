---
source_pdf: papers/Spectral Top-Down Recovery of Latent Tree Models.pdf
slug: spectral-top-down-recovery-of-latent-tree-models
pages: 45
extracted_on: 2026-07-13
---

# Spectral Top-Down Recovery of Latent Tree Models

## Page 1

Spectral Top-Down Recovery of Latent Tree Models
Yariv Aizenbud ∗1, Ariel Jaﬀe∗1, Meng Wang5, Amber Hu1, Noah Amsel1, Boaz
Nadler2, Joseph T. Chang3, and Yuval Kluger1,4,5
1Program in Applied Mathematics, Yale University, New Haven, CT 06511
2Department of Computer Science, Weizmann Institute of Science, Rehovot, 76100, Israel
3Department of Statistics, Yale University, New Haven, CT 06520, USA
4Interdepartmental Program in Computational Biology and Bioinformatics, Yale University, New Haven,
CT 06511
5Department of Pathology, Yale University New Haven, CT 06511
Abstract
Modeling the distribution of high dimensional data by a latent tree graphical model is a
prevalent approach in multiple scientiﬁc domains. A common task is to infer the underlying
tree structure, given only observations of its terminal nodes. Many algorithms for tree recovery
are computationally intensive, which limits their applicability to trees of moderate size. For
large trees, a common approach, termed divide-and-conquer, is to recover the tree structure in
two steps. First, recover the structure separately of multiple, possibly random subsets of the
terminal nodes. Second, merge the resulting subtrees to form a full tree. Here, we develop
Spectral Top-Down Recovery (STDR), a deterministic divide-and-conquer approach to infer
large latent tree models. Unlike previous methods, STDR partitions the terminal nodes in a
non random way, based on the Fiedler vector of a suitable Laplacian matrix related to the
observed nodes. We prove that under certain conditions, this partitioning is consistent with
the tree structure. This, in turn, leads to a signiﬁcantly simpler merging procedure of the small
subtrees. We prove that STDR is statistically consistent and bound the number of samples
required to accurately recover the tree with high probability. Using simulated data from several
common tree models in phylogenetics, we demonstrate that STDR has a signiﬁcant advantage
in terms of runtime, with improved or similar accuracy.
1
Introduction
Learning the structure of latent tree graphical models is a common task in machine learning [3, 10,
23, 42, 62] and computational biology [29, 30]. A canonical application is phylogenetics, where the
task is to infer the evolutionary tree that describes the relationship between a group of biological
species based on their nucleotide or protein sequences [18, 43, 48]. Depending on the application,
the number of observed nodes ranges from a dozen and up to tens of thousands.
In latent tree graphical models, every node is associated with a random variable. A key assump-
tion is that the given data corresponds to the terminal nodes of a tree, while the set of unobserved
∗YA and AJ contributed equally to this work
1
arXiv:2102.13276v2  [stat.ML]  7 Dec 2021

## Page 2

h1
h2
h3
h4
h5
x1
x2
x3
x4
x5
x6
x7
CCCCAAGGGGGATAGTAGTCAAA
CCACAAGGCGGCATACAGTCAAA
ACCCCAGGGCGATAGTGGTCAAA
ACGCAAGGCGGATAGCAGTCAAA
ACCTAAGGGCGATAGTCGTCAAA
ACTCAAGGGCGATAGTAGTCAAA
ACCCAAGGGGGATAGAAGTCAAA
Figure 1: A tree with m = 7 observed nodes. The data consists of a sequence of characters at every terminal
node.
internal nodes determines its distribution. In phylogenetics, the terminal nodes are existing organ-
isms, while the non-terminal nodes correspond to their extinct ancestors. Given a set of nucleotide
or amino acid sequences as in Figure 1, the task is to recover the structure of the tree, which
describes how the observed organisms evolved from their ancestors.
Many algorithms have been developed for recovering latent trees.
Distance-based methods,
including the classic neighbor joining (NJ) [46] and UPGMA [51], recover the tree based on a
distance measure between all pairs of terminal nodes. These methods are computationally eﬃcient
and thus applicable to large trees [56]. They also have statistical guarantees for accurate recovery
[4, 36]. Since the distance measure does not encapsulate all the information available from the
sequences, distance-based methods may perform poorly when the amount of data is limited [60].
A diﬀerent approach for tree recovery is based on spectral properties of the input data [2, 16].
Several methods work top-down, repeatedly applying spectral partitioning to the terminal nodes
until each partition contains a single node [35, 63]. However, there is no theoretical guarantee
that the partitions match the structure of the tree. Of direct relevance to this manuscript is the
recently proposed spectral neighbor joining (SNJ) [26], which consistently recovers the tree based
on a spectral criterion. Similarly to NJ, SNJ is a bottom-up method, which iteratively merges
subsets of nodes to recover the tree.
Perhaps one of the most accurate approaches for tree recovery is to search for the topology that
maximizes the likelihood of the observed data [18]. Since computing the likelihood for every possible
topology is intractable, many methods apply a local search to iteratively increase the likelihood
function [21, 44, 52, 64]. Though there is no guarantee that such a process will converge to the
global maximum of the likelihood function, in many settings the resulting tree is more accurate than
the one obtained by distance-based methods. The main disadvantage of likelihood-based algorithms
is their slow runtime, which limits their applicability to trees of moderate size.
With the dramatic increase in the sizes of measured datasets, there is a pressing need to develop
fast tree recovery algorithms, able to handle trees with tens of thousands of nodes [56, 47]. For exam-
ple, the recently developed GESTALT method combines scRNA-seq readouts with CRISPR/Cas9
induced mutations to perform lineage tracing on tens of thousands of cells.
[45, 50].
For the
multispecies coalescent model, recent works recover multiple gene trees, where each tree is com-
2

## Page 3

posed of thousands of genes [37]. Recently, many works recovered the evolutionary history of the
SARS-COV-2 virus, with over ten thousand variants [40].
Tree recovery problems with thousands of terminal nodes pose a signiﬁcant computational chal-
lenge, as even distance-based methods may prove to be too slow. To improve the scalability of
slow but accurate methods such as maximum likelihood, a common framework known as divide-
and-conquer is to recover the tree by a two-step process [39, 58]: (i) infer the tree structure inde-
pendently for a large number of small possibly random subsets of terminal nodes; (ii) compute the
full tree by merging the small trees obtained in step (i). In supertree methods, the small subsets of
terminal nodes in step (i) overlap. Their merging step requires optimizing a non-convex objective,
which is computationally hard [28, 25]. Thus, most supertree methods circumvent global optimiza-
tion problems by iterative approaches for step (ii) [55, 58]. Recently, several methods were derived
to merge subtrees with disjoint terminal nodes [39, 38]. To apply these algorithms in a divide-and-
conquer pipeline, the terminal nodes are partitioned according to an initial tree estimate computed
by NJ. Despite these works, the problem of reconstructing large trees from limited amount of data
is not yet fully resolved. In particular, there is still a need for fast and scalable approaches that
also have strong recovery guarantees.
Contributions and outline
In this work we develop Spectral Top-Down Recovery (STDR), a
scalable divide-and-conquer approach backed by theoretical guarantees to recover large trees. In
contrast to previous methods, the partitioning of the terminal nodes in step (i) is deterministic.
Importantly, we prove that under mild assumptions the partitions are consistent with the unob-
served tree structure.
The importance of this consistency is that it simpliﬁes considerably the
merging process in step (ii) of the algorithm. Since STDR is recursive, it is instructive to replace
the standard divide-and-conquer two step outline, with the following recursive description.
(i) Partitioning: split the terminal nodes into two subsets.
(ii) Recursive reconstruction: infer the latent tree of each subset. When the partition size falls
below a given threshold τ, the tree is recovered by a user-speciﬁed algorithm. Above this
threshold, the reconstruction is done by recursively applying STDR to each subset.
(iii) Merging: reconstruct the full tree by merging the two small trees.
Each of the above three steps is explained in detail in Section 3. In step (i) we apply spectral
partitioning to a weighted complete graph, with nodes that correspond to the terminal nodes of the
tree and weights based on a similarity measure described in Section 3.1. In Section 4.1 we prove
that given an accurate estimate of these similarities, step (i) is consistent in the sense that the
resulting subsets belong to two disjoint subtrees. For this proof, we derive a novel relation between
latent tree models and a classic result from spectral graph theory known as Fiedler’s theorem of
nodal domains [19]. This theorem is important in various learning tasks such as clustering data
[57], graph partitioning [12], and low dimensional embeddings [27]. To the best of our knowledge,
this is the ﬁrst guarantee for spectral partitioning in the setting of latent tree models.
The output of step (ii) is the inner structure of two disjoint subtrees. The task in step (iii) is to
merge them into the full tree. In Section 3.4, we show that this task is equivalent to ﬁnding the root
of an unrooted tree, given a reference set of one or more sequences, also known as an outgroup. We
derive a novel spectral-based method to ﬁnd the root and prove its statistical consistency in Section
4.2. This approach is of independent interest, as ﬁnding the root of a tree is a common challenge
3

## Page 4

in phylogenetics [6, 8, 32]. Finite sample guarantees for the Jukes-Cantor model of evolution are
derived in Section 5.
In Section 6 we compare the accuracy and runtime of various methods when applied to recover
the full tree directly versus when used as subroutines in step (ii) of STDR. For example, Figure
6 shows the results of recovering simulated trees with 2000 terminal nodes generated according
to the coalescent model [49]. As one baseline, we applied RAxML [52], one of the most popular
maximum likelihood software packages in phylogenetics. With 8,000 samples, RAxML took over
5 1
2 hours to complete. In contrast, STDR with RAxML as subroutine and a threshold τ = 128 took
approximately 21 minutes, more than an order of magnitude faster. Importantly, in this setting,
the trees recovered via STDR have similar accuracy to those obtained by applying RAxML directly.
These and other simulation results illustrate the potential beneﬁt of STDR in recovering large trees.
2
Problem setup
Let T be an unrooted binary tree with m terminal nodes.
We assume that each node of the
tree has an associated discrete random variable over the alphabet {1, . . . , ℓ}. We denote by x =
(x1, . . . , xm) the vector of the random variables at the m observed terminal nodes of the tree, and by
h = (h1, . . . , hm−2) the random variables at the non-terminal nodes. We assume that these random
variables form a Markov random ﬁeld on T . This means that given the values of its neighbors, the
random variable at a node is statistically independent of the rest of the tree [9]. An edge e(hi, hj)
connecting a pair of adjacent nodes (hi, hj) is equipped with two transition matrices of size ℓ× ℓ,
P(hi|hj)ba = Pr[hi = b|hj = a],
P(hj|hi)ba = Pr[hj = b|hi = a].
(1)
Note that every pair of adjacent nodes may in general have diﬀerent transition matrices.
Our observed data is a matrix X = [x(1), . . . , x(n)] ∈{1, . . . , ℓ}m×n, where x(j) are random i.i.d.
realizations of x = (x1, . . . , xm). Each row in the matrix is a sequence of length n that corresponds
to a terminal node in the tree, see illustration in Figure 1. For example, in phylogenetics, each
row in the matrix corresponds to a diﬀerent species, while each column corresponds to a diﬀerent
location in a DNA sequence, see [14] and references therein. Figure 1 shows an example with m = 7
terminal nodes and n = 23 observations. The support of each node is the DNA alphabet A, C, G, T,
so ℓ= 4.
Given the matrix X, the task at hand is to recover the structure of the hidden tree T . We
assume that for every pair of adjacent nodes (hi, hj), the corresponding ℓ× ℓstochastic matrices
P(hi|hj) and P(hj|hi) deﬁned in (1) are full rank, with determinants that satisfy
0 < δ < det(P(hi|hj)), det(P(hj|hi)) < ξ < 1.
(2)
Eq. (2) implies that the transition matrices are invertible and are not permutation matrices. This
assumption is necessary for the tree’s topology to be identiﬁable, see Proposition 3.1 in [9] and [41].
Next, to describe our approach we present several deﬁnitions related to unrooted trees, following
the terminology of [59].
Deﬁnition 1 (clan). A clan is a subset of nodes in T that is connected to the rest of the tree by a
single edge.
Deﬁnition 2 (the root of a clan). A non-terminal node h is termed the root of a clan C if h ∈C
and it is connected to the edge that separates C from the rest of the tree.
4

## Page 5

For example, in Figure 1 h4 and h5 are the root nodes of the clans C1 = {x6, x7, h5} and
C2 = {x4, x5, x6, x7, h2, h4, h5}, respectively. In our work, we will sometimes refer to the clans by
their terminal nodes only (e.g. {x6, x7} and {x4, x5, x6, x7} for C1 and C2).
Deﬁnition 3 (adjacent clans). Let C1 and C2 be two disjoint subsets of terminal nodes that form
two clans. If the union C1 ∪C2 forms a clan, then C1 and C2 are adjacent clans.
Two disjoint clans whose respective root nodes share a common neighboring node are adjacent
clans. For example, in Figure 1 the clans C1 = {x4, x5} and C2 = {x6, x7} are adjacent. Their
respective root nodes h4 and h5 are adjacent to h2. This observation is important for the merging
step of STDR.
3
A spectral top-down approach for tree reconstruction
Here we present the three steps of the Spectral Top-Down Recovery (STDR) algorithm, as outlined
in the introduction.
Pseudocode for the method appears in Algorithm 1.
We begin with the
deﬁnition and properties of the similarity matrix and similarity graph.
3.1
The pairwise similarity matrix and similarity graph
Similar to Eq. (1), we deﬁne the ℓ× ℓtransition matrix for every pair hi, hj of (not necessarily
adjacent) nodes by
P(hi|hj)ba = Pr[hi = b|hj = a].
Note that due to the Markov assumption, the transition matrix is multiplicative along the edges
of the tree. For example in Figure 1, P(x1|x2) = P(x1|h3)P(h3|x2). In [26], a similarity function
between a pair of nodes hi and hj was deﬁned as follows:
S(hi, hj) =
q
det(P(hi|hj)) det(P(hj|hi)).
(3)
Similar to the transition matrix, the similarity is multiplicative along the edges of the tree and is
bounded by δ ≤S(hi, hj) ≤ξ. Thus, it exhibits an exponential decay along the tree. For any two
ordered sets of terminal or non-terminal nodes A = {a1, . . . ar} and B = {b1, . . . bs}, we denote by
S(A, B) a matrix of size r × s, where
S(A, B)ij = S(ai, bj)
for all 1 ≤i ≤r and 1 ≤j ≤s.
To simplify notation, for the case where A and B are both equal to the full set of terminal nodes,
we denote the similarity matrix by S:
S = S(x, x)
where x = {x1, . . . , xm}.
(4)
where by deﬁnition, Sii = 1 ∀(i). The matrix S is the adjacency matrix of the following graph.
Deﬁnition 4 (Similarity graph). The similarity graph G is a complete graph whose vertices are
the terminal nodes of T . The weight assigned to every edge e(xi, xj) is the similarity S(xi, xj).
The relation between the spectral properties of G and the topology of T forms the theoretical
basis of our approach. The following result from [26, Lemma 3.1] shows how the spectral structure
of the similarity matrix S relates to the structure of the underlying tree.
5

## Page 6

Algorithm 1 STDR: Spectral Top-Down Recovery
Input: X ∈{1, . . . , ℓ}m×n
A matrix containing sequences from m terminal nodes
τ ∈N
Partition size threshold
Alg
An algorithm for recovering small tree structures
Output: T
Estimated tree
1: if number of terminal nodes m ≤τ then
2:
return Alg(X)
▷Recover small tree structures by a user deﬁned algorithm
3: end if
4: Compute the similarity matrix S from X via Eq. (4)
5: Compute the Fiedler vector v of S
▷Partitioning step
6: Partition the terminal nodes into two subsets C1 and C2 by thresholding v via Eq. (5)
▷Recursive reconstruction step
7: T1 = STDR(X(C1, :), τ, Alg)
8:
T2 = STDR(X(C2, :), τ, Alg)
▷Merging step
9: Compute u, the ﬁrst left singular vector of S(C1, C2)
10: for all edges e in T1 do
11:
Compute the edge score d(e) from u via Eq. (8)
12: end for
13: Insert a root node for T1 into the edge e1 = argmine∈T1 d(e)
14: Compute v, the ﬁrst right singular vector of S(C1, C2)
15: for all edges e in T2 do
16:
Compute the edge score d(e) from v via Eq. (8)
17: end for
18: Insert a root node for T2 into the edge e2 = argmine∈T2 d(e)
19: Connect the roots of T1 and T2 to construct the merged tree T
20: return T
Lemma 3.1. Let A and B be a partition of the terminal nodes of an unrooted binary tree T . The
matrix S(A, B) is rank-one if and only if A and B are clans of T .
Lemma 3.1 implies that given the exact similarity matrix S, one can determine if a subset A of
terminal nodes is a clan in T by computing the rank of S(A, Ac), where Ac = x \ A. In practice,
the exact similarity matrix S is unknown. Yet, as shown in [26], a suﬃciently accurate estimate ˆS,
which in general is full rank, still allows to determine if a subset is a clan.
3.2
Tree partitioning via spectral clustering
The aim of step (i) of STDR is to partition the terminal nodes into two clans of T . Our approach
is based on the similarity graph G of Deﬁnition 4. One possible way to partition the graph is by
the min-cut criteria. Given the exact similarity, this approach is guaranteed to yield two clans, see
Lemma B.1 in the appendix. Though the min-cut problem can be solved eﬃciently [57], it often
leads to unbalanced partitions of the graph, with the smaller one containing 1 or 2 terminal nodes.
6

## Page 7

(a) Illustration of a symmetric binary tree.
0
20
40
60
80
100
120
−0.10
−0.05
0.00
0.05
0.10
(b) Fielder vector of binary symmetric tree
Figure 2: Symmetric binary tree with 128 terminal nodes. The data consists of sequences of length n = 1000
over the ℓ= 4 characters of the DNA alphabet, generated according to the HKY model.
Since one goal is to reduce the runtime of the reconstruction algorithm in step (ii), we would like to
avoid imbalanced partitions. To this end, we propose to partition the terminal nodes via a spectral
approach based on the Fiedler vector.
Deﬁnition 5 (Graph Laplacian and Fiedler vector). The Laplacian matrix of a graph G with a
symmetric weight matrix W is given by LG = D −W, where D is a diagonal matrix with Dii =
P
j W(xi, xj). The Fiedler vector is the eigenvector of LG that corresponds to the second smallest
eigenvalue.
In the STDR algorithm, we use the Fiedler vector v of the similarity graph G to partition the
terminal nodes into two subsets C1 and C2 (Algorithm 1, line 6), as follows:
C1 = {i; v(i) ≥0},
C2 = {i; v(i) < 0}.
(5)
Importantly, in Section 4.1 we prove that partitioning the nodes of G via Eq. (5) yields two clans of
the underlying tree T . To illustrate this point, we created a tree graphical model from a symmetric
binary tree with m = 128 nodes, see Figure 2a. The transition matrices between adjacent nodes
are all identical and were chosen according to the HKY model [24]. We used this model to generate
a dataset of nucleotide sequences of length n = 1, 000. Figure 2b shows the Fiedler vector of the
similarity graph estimated from the dataset. Here, the Fiedler vector exhibits a single dominant
gap, and partitioning the terminal nodes by Eq. (5) yields two sets C1 and C2 which are indeed
clans of T . A similar example is shown in the appendix for a tree generated according to the
coalescent model. In Section 5.1 we derive an expression for the number of samples required to
obtain two clans with high probability.
3.3
Recursive Reconstruction Step
Step (i) of STDR outputs two sets of terminal nodes C1 and C2. Under certain conditions deﬁned
in Section 4.1, these are guaranteed to be two clans in the tree T . The next task is to construct
trees T1 and T2 that describe their latent internal structure. If |C1| > τ, then T1 is recovered by
7

## Page 8

h1
hA
hB
x1
x2
x3
x4
h2
hC
x5
x6
x7
(a) Two unrooted trees. The placeholder edges are
marked in red.
h1
hA
hB
x1
x2
x3
x4
h2
hC
x5
x6
x7
(b) The merging process is completed by connecting
the two root nodes h1, h2.
Figure 3: Merging example
recursively reapplying the three steps of STDR to C1. When |C1| ≤τ, the input is small enough
that we consider it tractable to use a direct method for tree reconstruction, even a slow one like
maximum likelihood.
3.4
Merging disjoint subtrees
The output of step (ii) of STDR consists of the internal unrooted tree structures T1 and T2 of two
subsets of terminal nodes C1 and C2. Assuming steps (i) and (ii) were successful, then C1 and
C2 are adjacent clans, and T1 and T2 are indeed their correct internal structure. The remaining
challenge in step (iii) is to recover the full tree T by correctly merging T1 and T2.
Since T1 and T2 are unrooted binary trees, to merge them it is necessary to add a root node
to each of them. Adding a connecting edge between the two root nodes yields a binary unrooted
tree and completes the merging process, see Figure 3 for an illustration. To add a root node to
a subtree, we select one of its edges to be the “placeholder edge” (illustrated in red in Figure
3a). Subsequently, the placeholder edge is replaced with two edges connected to the root node.
Importantly, as shown in Figure 4, changing the placeholder edge in either T1 or T2 yields a merged
tree with a diﬀerent topology.
Thus, merging T1 and T2 reduces to the task of identifying the correct “placeholder edge”.
Here, we derive a novel spectral method for ﬁnding these edges. To the best of our knowledge, our
approach for merging subtrees is new and may be of independent interest for other applications,
such as rooting unrooted trees [32, 8, 6]. In the following lemma, whose proof is in Appendix C,
we describe a property of the placeholder edge that motivates our approach.
Lemma 3.2. Let C1 be a set of terminal nodes that forms a clan in T , and let T1 be the internal
structure of C1. An edge e ∈T1 is the correct placeholder edge if and only if it partitions C1 into
two sets A(e), B(e), such that both form clans in T .
Lemma 3.2 is illustrated in Figure 3a. The edge e(hA, hB) divides the left subtree into the clans
{x1, x2} and {x3, x4}. These subsets also form clans in the full tree depicted in Figure 3b. Next,
using Lemma 3.2, we derive a spectral characterization of the correct placeholder edge. Recall that
by Lemma 3.1, the matrix S(C1, C2) ∈R|C1|×|C2|, is rank one. Thus,
S(C1, C2) = uσvT ,
where
∥v∥= ∥u∥= 1,
and
σ > 0.
(6)
8

## Page 9

h1
hA
hB
x1
x2
x3
x4
h2
hC
x5
x6
x7
(a) Placeholder edges set to e(hA, hB) and e(hC, x5).
h1
hA
hB
x1
x2
x3
x4
h2
hC
x5
x6
x7
(b) Placeholder edges set to e(hB, x4) and e(hC, x5).
Figure 4: Diﬀerent choices of placeholder edges result in a diﬀerent merged trees.
Given a placeholder edge e and its corresponding partition of terminal nodes A(e) and B(e), we
denote by uA(e), uB(e) the entries of u that correspond to A(e) and B(e), respectively. The following
lemma, proven in Appendix C, characterizes the correct placeholder edge in terms of uA(e) and uB(e).
Lemma 3.3. An edge e is the correct placeholder edge of T1 if and only if there exists a constant
α such that
S(A(e), B(e)) = uA(e)αuT
B(e).
(7)
In practice we can only compute an estimate of S. Motivated by Lemma 3.3, we propose to
determine the placeholder edge e∗by minimizing the following score function,
e∗= argmin
e
d(e) = argmin
e
1
∥S(A(e), B(e))∥F
min
α ∥S(A(e), B(e)) −uA(e)αuT
B(e)∥F .
(8)
The normalizing factor ∥S(A(e), B(e))∥F is added since the size of S(A(e), B(e)) changes for every
edge e. Note that given the exact matrix S, at the correct placeholder edge d(e∗) = 0. In Section
5.2 we derive an expression for the number of samples required to obtain the correct placeholder
edge by Eq. (8) with high probability.
3.5
Computational complexity
We analyze the complexity of each step of STDR separately. We assume that the similarity or
distance matrix are given. To simplify the analysis, we assume a balanced binary tree, and that the
partition steps gave m/τ subsets of size τ each. We denote by B(k) the complexity of recovering
the topology of a tree with k terminal nodes by the given subroutine Alg.
1. Given the similarity matrix, partitioning a set of k terminal nodes is O(k2), due to the
computation of the Fiedler vector of the positive semi-deﬁnite Laplacian matrix [53, Chapter
2].
2. The complexity of merging two subtrees with k terminal nodes each is composed of two parts:
(i) compute the leading singular vector of the matrix S(C1, C2) ∈Rk×k, which takes O(k2)
operations; (ii) compute the score for every edge as in Eq. (8). The number of operations
required for the least square operation in the numerator of Eq. (8), as well as computing the
Frobenius norms in the numerator and denominator is proportional to the number of elements
9

## Page 10

in S(A(e), B(e)). Thus, the total complexity of computing the score for all edges in T1 (and
similarly T2) is O(P
e∈T1 |A(e)||B(e)|). For a balanced tree, this term is equal to
k2
4 +
log k
X
i=2
2i
|{z}
Number of partitions
of size k/2i
k
2i
|{z}
|A(e)|

k −k
2i

|
{z
}
|B(e)|
= O(k2 log k).
We remark that if the two trees are highly imbalanced, the complexity may increase up to
O(k3).
Let T(m) be the complexity of the partitioning and merging operations of STDR, excluding the
complexity of the subroutine algorithm that recovers the structure of small trees. We have that
T(m) =
O(m2)
| {z }
partitioning
+2T(m/2) + O(m2 log m)
|
{z
}
merging
= 2T(m/2) + O(m2 log m).
By the Master theorem [7],
T(m) = O(m2 log m).
(9)
Thus, the total complexity of STDR is
O(m2 log m + (m/τ)B(τ)).
For example, the complexity of NJ is B(τ) = O(τ 3).
Thus, the complexity of STDR+NJ is
O(m2 log m + mτ 2), which for τ = O(1) improves upon the O(m3) complexity of running NJ to
recover the full tree. In the simulation section, we show that STDR+NJ outperforms NJ in accuracy
while being about an order of magnitude faster.
An important property of the STDR algorithm, in terns of actual runtime, is that it is embar-
rassingly parallel. Speciﬁcally, steps 7 and 8 in Algorithm 1 can be executed in two independent
processes. This may result in up to k parallel processes, where k is the number of partitions.
4
Correct tree recovery of STDR
In this section we consider the population setting where the similarity matrix S is known. In this
setting we prove that STDR correctly recovers the underlying tree. We do so by analyzing the
partitioning step (i) and the merging step (iii) of STDR. Our key results are Theorem 4.2, which
states that step (i) is guaranteed to yield disjoint clans, and Theorem 4.5, which states that given
accurate trees for two clans, step (iii) recovers the exact structure of the full tree.
Combining
these two results directly yields the following theorem establishing the correctness of STDR in the
population setting.
Theorem 4.1. Given an exact similarity matrix S, and assuming that the subroutine Alg correctly
recovers the internal structure of its input, STDR recovers the exact latent tree T .
10

## Page 11

4.1
Consistency of the partition step
The following theorem proves that given the exact similarity matrix, partitioning the terminal nodes
of the tree by thresholding the Fiedler vector as described in Section 3.1 yields two adjacent clans.
Theorem 4.2. Let G be the similarity graph of a binary tree T . Denote by v the Fiedler vector
of G and by {C1, C2} a partition of the terminal nodes according to the sign pattern of v as in Eq.
(5). Then C1, C2 are adjacent clans in T .
Before proving Theorem 4.2, we would like to put its novelty into the context of related results.
In A result similar in nature to Theorem 4.2 was proved for hierarchical block models (HBM)
[5] where the underlying block structure of a given connectivity matrix is recovered by recursive
partitioning according to its Fiedler vector. The statistical guaranty, however, is derived by making
additional assumptions on the structure of the tree as well as its parameters. Theorem 4.2, in
contrast, is true for any tree structure and parameters. A diﬀerent distance-based approach for tree
partitioning was derived in chapter 4 of [20]. This approach is guaranteed to yield two clans, but
only given the exact distance matrix between terminal nodes. In Appendix D we show empirically
that our similarity based approach is more robust than the distance based approach, speciﬁcally in
cases where the number of samples is limited.
For the proof of Theorem 4.2, we present several preliminaries on graphs. First, we deﬁne the
Schur complement of a matrix, which plays an important role in graph theory [13].
Deﬁnition 6 (Schur complement). Let A, B, C and D be matrices of dimensions p × p, p × q, q × p
and q × q, respectively. Assume D is invertible and consider the matrix
M =

A
B
C
D

,
of size (p + q) × (p + q). The Schur complement of M with respect to D is the p × p matrix
M/D = A −BD−1C.
Let H be a graph with a set of nodes V and Laplacian matrix L. We denote by LR the principal
sub-matrix of L that corresponds to a subset of nodes R ⊂V . The Schur complement of L with
respect to LR yields the Laplacian of a diﬀerent graph, with |V −R| nodes [11, 13]. We denote this
Laplacian matrix by LH/R. The rows and columns of LH/R correspond to vertices of H that are
not in R. When the graph is a tree T , and R is the set of its non-terminal nodes, then LT /R is the
Laplacian of a complete graph G whose nodes are the terminal nodes of T .
Equipped with these deﬁnitions, we proceed to the proof of Theorem 4.2. The proof consists of
two parts, that correspond to Theorem 4.3 and Lemma 4.4. Theorem 4.3, which is a rephrase of
Theorem 3.3 of [54], shows that one can partition the terminal nodes of a tree T into two clans via
the Fiedler vector of LT /R, where R is the set of all internal nodes.
Theorem 4.3 ([54],Theorem 3.31). Let T be a tree with a node set V and a subset of non terminal
nodes R ⊂V . We denote by LT the Laplacian of T and by LT /R the Laplacian of a graph G
obtained by Schur complement of LT with respect to R. Let v be the Fiedler vector of G, and C1, C2
the following partition of the terminal nodes,
C1 = {i ∈V \ R; v(i) ≤0},
C2 = {j ∈V \ R; v(j) > 0}.
Then C1 and C2 are adjacent clans in T .
1For clarity, we rephrased the theorem from [54] according to our terminology.
11

## Page 12

Theorem 4.3, however, is not directly applicable to our setting, since computing LT /R requires
knowledge of the unknown similarities between all nodes of T , including its unobserved nodes.
Here, we derive Lemma 4.4 that shows that for any tree T , there is a twin tree eT with the same
topology, such that L e
T /R = LG. This result, proven in appendix E, provides the critical missing
link required for inference of the latent tree from the similarity matrix, which can be estimated
from observed data.
Lemma 4.4. Let T be a tree with a set of non-terminal nodes R. Let G be the similarity graph of
T . Then there is a tree eT with the same topology as T but diﬀerent edge weights, such that
LG = L e
T /R.
Combining Lemma 4.4 with Theorem 4.3 yields the following proof of Theorem 4.2.
Proof of Theorem 4.2. Let LG be the Laplacian matrix of the similarity graph G. By Lemma 4.4
there is a tree eT with the same topology as T such that LG = L e
T /R. By Theorem 4.3, partitioning
the terminal nodes of eT according to the sign pattern of the Fiedler vector of L e
T /R yields adjacent
clans in eT . Since LG = L e
T /R and eT has the same topology as T , it follows that partitioning the
terminal nodes of T according to the Fiedler vector of LG yields adjacent clans in T .
4.2
Correctness of the merging step
Step (iii) of STDR merges the two subtrees, T1 and T2, that were constructed from the two disjoint
subsets of terminal nodes C1 and C2. As described in Section 3, this step is done by ﬁnding for
each tree its placeholder edge as the edge with the smallest score d(e), Eq. (8). Here, we prove that
this merging step is correct, under the following two assumptions on its input (the output of steps
(i) and (ii)): the two subtrees T1, T2 correspond to adjacent clans in T and their internal structure
was recovered correctly.
Theorem 4.5. Let C1 and C2 be the terminal nodes of two adjacent clans that partition a tree
T . Let T1 and T2 be the internal structures of these clans. Then given the exact similarity matrix
S(C1, C2), minimizing the criterion in Eq. (8) yields the correct placeholder edge.
Proof. By Lemma 3.3, for the correct placeholder edge e∗there exists an α ∈R such that
S(A(e∗), B(e∗)) = uA(e∗)αuT
B(e∗).
Hence d(e∗) = 0. If e is an incorrect placeholder edge, then again according to Lemma 3.3 there is
no constant α that satisﬁes the equation, and hence d(e) > 0 which implies e∗= argmin d(e).
5
Finite sample guarantees for STDR
In practice, the true similarity matrix S is unknown, and an estimate ˆS is computed from a sequence
data of length n. In this section we show that STDR is still able to correctly recover the tree provided
that ˆS is suﬃciently close to S. Speciﬁcally, in sections 5.1 and 5.2, we derive lower bounds on
the number of samples required for the partitioning step and the merging step to succeed with
high probability. In Section 5.3 we compare these results to the guarantees available for other tree
recovery algorithms.
12

## Page 13

For simplicity, in the ﬁnite sample analysis, we assume the Jukes-Cantor (JC) model of sequence
evolution, where each transition matrix is parameterized by a single mutation rate θ(i, j):
P(hi|hj)ba = P[hi = b|hj = a] =
(
1 −θ(i, j)
a = b
θ(i, j)/(ℓ−1)
a ̸= b.
(10)
According to this model, the similarity between adjacent nodes deﬁned in Eq. (4) simpliﬁes to
S(hi, hj) =

1 −
ℓ
ℓ−1θ(i, j)
ℓ−1
.
By Eq. (2) the similarity is strictly positive and hence θ(i, j) < (ℓ−1)/ℓ. We remark that our
analysis can be extended, under minor additional assumptions to more general models of evolution
as in [26, Lemma 4.8]. We present results for the top level of the tree partitioning and merging.
Following the proof, we show in Remark 5.10 that the same guarantees hold for multiple partitions
and merging steps.
5.1
Finite sample guarantees for the partitioning step
We compute the number of samples n required for the partitioning step to yield two clans with
high probability. To this end, we require that in the population setting, the entries of the Fiedler
vector are bounded away from zero. To that end, we assume that the similarity matrix S satisﬁes
the hierarchical constant block model (CBM) addressed in [5]. We assume there is a hierarchy of
partitions AC and BC, such that for each partition there is a (diﬀerent) constant c such that
S(x, y) = c
∀(x, y) ∈AC × BC,
S(x, y) > c
∀(x, y) ∈AC × AC
and
∀(x, y) ∈BC × BC.
(11)
In phylogenetics, this assumption is satisﬁed in the molecular clock model [34], where the probability
of mutation between adjacent nodes is determined by two factors: (i) the edge length between them
and (ii) a mutation rate matrix that is constant throughout the tree. The structure of the rate
matrix is determined by the choice of evolutionary model, such as Jukes-Cantor. In addition, the
path length between all terminal nodes and the root is constant. This implies that for every ancestor
h (internal node) the similarity between the terminal nodes AC on the left of h and the nodes BC
on the right of h is constant as in Eq. (11). For the hierarchy of partitions, we denote by η the
maximum over all partitions C of the ratio between the size of left and right parts AC, BC.
η = max
C {|AC|/|BC|, |BC|/|AC|}.
(12)
This factor serves as a measure for the balancedness of the tree. In addition, we denote by r(T )
the diameter of T , which is the maximal distance between pairs of terminal nodes,
r(T ) = max
i,j (−log S(xi, xj)).
(13)
Finally, we denote by h(T ) the depth of T as deﬁned in [15]:
13

## Page 14

Deﬁnition 7. Let T1, T2 be two rooted subtrees with respective roots h1, h2 obtained by removing an
edge e(h1, h2) from T . Let d1(e), d2(e) be the distances log S(h1, xi) and log S(h2, xj) from h1, h2
to the closest leaves xi and xj in T1, T2, respectively. Then
h(T ) = max
e
max{d1(e), d2(e)}.
(14)
Note that h(T ) < r(T ) as the maximal distance between terminal nodes is larger than any
distance between a pair of terminal and non terminal nodes. The following theorem bounds the
number of samples n by the properties of the tree deﬁned in Eqs. (12),(14) and (14).
Theorem 5.1. Let T be a Jukes-Cantor evolutionary tree with m terminal nodes, with a similarity
matrix S that satisﬁes the assumptions made for the CBM. If the number of samples n satisﬁes
n ≥4 ln
2m2
ϵ

ηℓ2m(√m + 1)2e2r(t) max

1,
(1 + η)2
(er(T )−h(T ) −1)2

,
then STDR partitions the terminal nodes into two clans with probability at least 1 −ϵ.
To prove the theorem, we derive a bound on the error that the partitioning step can tolerate in
the estimate ˆS.
Lemma 5.2. Assume a tree with m terminal nodes generated according to the molecular clock
model. If the estimate ˆS of its similarity matrix satisﬁes
∥S −ˆS∥≤
√me−r(T )
√η23/2(√m + 1) min

1,
1
1 + η
 er(T )−h(T ) −1

,
(15)
then STDR correctly partitions the terminal nodes into two clans.
In our proof, we use the following lemma regarding the spectrum of the Laplacian. This Lemma
is a reformulation of lemma 7 from [5] that addresses the spectrum of the CBM.
Lemma 5.3. Consider a tree with m terminal nodes generated according to the molecular clock
model. Let L be the Laplacian of its similarity graph. The ﬁrst second and third smallest eigenvalues
of L satisfy
λ1 = 0,
λ2 = me−r(T ),
λ3 ≥
m
1 + η
 ηe−r(T ) + e−h(T )
.
The elements v2(i) of the eigenvector that corresponds to λ2 satisfy |v2(i)| ≥
q
1
mη.
Proof of Lemma 5.2. Let L and ˆL be two symmetric matrices and let vi and ˆvi be their i-th eigen-
vectors, respectively. A variant of the Davis-Kahan theorem for perturbation of eigenvectors (see
Theorem 2 of [61]) gives
∥vi −ˆvi∥≤23/2 ∥L −ˆL∥
γi
.
(16)
where γi = min{|λi −λi+1|, |λi −λi−1|} is the eigengap. We apply the theorem to the Laplacian
matrix L = D −S (see Deﬁnition 5), and its Fiedler vector v2. The spectral norm ∥L −ˆL∥can be
14

## Page 15

bounded by,
∥L −ˆL∥≤∥D −ˆD∥+ ∥S −ˆS∥= max
i

X
k
(Sik −ˆSik)
 + ∥S −ˆS∥
≤max
i
X
k
|Sik −ˆSik| + ∥S −ˆS∥≤(√m + 1)∥S −ˆS∥.
(17)
Substituting (17) into (16) yields
∥v2 −ˆv2∥≤23/2(√m + 1)∥S −ˆS∥
γ2
.
(18)
From Lemma 5.3 it follows that the spectral gap γ2 is bounded by,
γ2 = min(λ2 −λ1, λ3 −λ2) ≥me−r(T ) min

1,
1
1 + η
 er(T )−h(T ) −1

.
(19)
Combining Eqs. (18) and (19) proves that if
∥S −ˆS∥≤
√me−r(T )
√η23/2(√m + 1) min

1,
1
1 + η
 er(T )−h(T ) −1

(20)
then ∥v2 −ˆv2∥< 1/√ηm, which implies ∥v2 −ˆv2∥∞< 1/√ηm. Thus, by Lemma 5.3 sign(vi) =
sign(ˆvi) for each i ∈[m]. Hence, partitioning the terminal nodes according to sign(ˆv2) or sign(v2)
yield the same result. As we proved in Theorem 4.2, the resulting subsets are clans of the tree.
Next, we prove Theorem 5.1 under the additional assumption of the Jukes-Cantor model. The
theorem is proved by combining Lemma 5.2 with a concentration bound on the similarity matrix
estimate ˆS, derived in [26].
Proof of Theorem 5.1. From Lemma 4.7 of [26], under the JC model of evolution,
P

∥ˆS −S∥≤t

≥1 −2m2exp

−2nt2
ℓ2m2

.
Setting t to the right hand side of (20) yields that if
n ≥4 ln
2m2
ϵ

ηℓ2m(√m + 1)2e2r(t) max

1,
(1 + η)2
(er(T )−h(T ) −1)2

,
the requirements of Lemma 5.2 are satisﬁed with probability at least 1 −ε, which concludes the
proof.
5.2
Merging step of STDR
We derive ﬁnite sample bounds for the merging step of STDR. In contrast to the partitioning step,
the guarantees for the merging step, presented in the following theorem, hold for any tree topology.
15

## Page 16

Theorem 5.4. Let T be a tree with m terminal nodes, which consists of two subtrees T1, T2 with
terminal nodes C1 and C2, respectively. Let {A, B} be the partition of C1 induced by the correct
placeholder edge e∗, and let D = min{∥S(A, B)∥F , ∥S(C1, C2)∥F }. For any ε > 0, if the number of
samples n satisﬁes
n ≥8ℓ2m3
 
2
D + 2.5
D2 + 1 + 10
√
2
D3
!2 
ξ4
δ6(1 −ξ2)2

log
2m2
ε

,
(21)
then STDR ﬁnds the correct placeholder edge in T1 with probability at least 1 −ε.
From Eq. (21), if D ≫1 then the required number of samples is eO(m3/D2). Assuming that the
lower bound on the similarity between adjacent nodes δ is close to 1, the value of D depends mainly
on the size of the two submatrices S(A, B) and S(C1, C2). This analysis has important implications
on the choice of the smallest partition τ in Algorithm 1. The number of samples in Eq. (21) is
eO(m) if A, B and C2 are of size O(m), but is eO(m3) if A and B or C2 are of size O(1). Thus on
the one hand, reducing τ results in smaller subsets of terminal nodes, which improves the runtime
of the reconstruction step of STDR. On the other hand it may aﬀect the accuracy of the merging
step. Figure 8 shows both runtime and accuracy of STDR as a function of the threshold parameter
τ, when applying STDR with RAxML or SNJ as its subroutine. The data consists of n = 1000
samples generated from a binary symmetric tree with m = 2048 terminal nodes. The accuracy of
the algorithm degrades for small values of τ while the runtime improves by approximately half an
order of magnitude.
Our proof of Theorem 5.4 consists of three steps: (i) In Lemma 5.5 we derive a lower bound
on the score d(e) of an edge e that is not the correct placeholder edge. (ii) Lemma 5.8 provides a
suﬃcient condition on the accuracy of the similarity matrix estimate ˆS that guarantees the merging
step will yield the correct placeholder edge. (iii) For the JC model, we derive an expression for the
number of samples required for the condition in Lemma 5.8 to hold with high probability.
Step 1: A lower bound on the score d(e) for incorrect edges
In Section 4.2, we showed
that d(e) = 0 if and only if e is the correct placeholder edge. Here, for the exact similarity matrix
S we derive a lower bound on d(e), if e is an incorrect placeholder edge in T1.
Lemma 5.5. Let T be a tree that consists of two subtrees T1, T2, and let e ∈T1 be an edge that is
not the correct placeholder edge. Then,
d(e) ≥



(
√
2δ)log mδ2(1−ξ2)
2√mξ2
δ2 ≤0.5
δ3(1−ξ2)
√
2mξ2
δ2 > 0.5.
For the proof of Lemma 5.5, we introduce new notation, illustrated in Figure 5. The sets of
terminal nodes of T1 and T2 are denoted by C1 and C2, respectively. We denote by e∗∈T1 the
correct placeholder edge, and by e ∈T1 an arbitrary incorrect placeholder edge. The edge e splits
the terminal nodes of T1 into A and B and has endpoints hA and hB. We denote by h0, . . . , hN
the non terminal nodes on the path between the root node of T1, denoted h0, and hA = hN. We
partition the terminal nodes in A to N + 1 subsets A0, . . . , AN according to h0, . . . , hN as follows:
Every node in A is assigned to the closest non terminal node on the path between h0, . . . , hN. In
the proof of Lemma 5.5, we use the following auxiliary lemma, proven in appendix F.
16

## Page 17

A0
A1
A2
B
h0
h1
hA
hB
x1
x2
x3
x4
x5
x6
x7
x8
e
e∗
T2
Figure 5: Bounding the score d(e) for an incorrect placeholder edge in T1. The correct placeholder edge
e∗∈T1 is marked by a dotted blue line. The incorrect placeholder edge e, which partitions the terminal node
to subsets A(e) and B(e), is marked by a thick red line. The two non-terminal nodes on the path between
the correct and incorrect edges are denoted by h1, h2 = hA, and the root node of C1 is denoted by h0. The
subset of terminal nodes closest to hi is denoted by Ai.
Lemma 5.6. Let Ri = S(h0, hi)2. For any 1 ≤i ≤N −1 and 1 ≤k ≤(N −i) we have
min
β
(1 −βRi)2∥S(Ai, B)∥2
F + (1 −βRi+k)2∥S(Ai+k, B)∥2
F
∥S(Ai, B)∥2
F + ∥S(Ai+k, B)∥2
F
≥
( (2δ2)log mδ2(k+1)(1−ξ2)2
4mξ4
δ2 ≤0.5
δ2(k+2)(1−ξ2)2
2mξ4
δ2 > 0.5
(22)
Proof of Lemma 5.5 . The proof consists of the following steps: (i) we rewrite the score d(e) deﬁned
in Eq. (8) in terms of ∥S(A0, B)∥F , . . . ∥S(AN, B)∥F . The new expression is given in Eq. (26).
(ii) In Eq. (27) we derive a lower bound on d(e) in terms of two consecutive terms ∥S(Ai, B)∥F
and ∥S(Ai+1, B)∥F . (iii) In Lemma 5.6 we combine Eq. (27) with a bound on ∥S(Ai, B)∥F and
∥S(Ai+1, B)∥F to conclude the proof.
First, we express the numerator of d(e) in Eq. (8) in terms of S(A0, B), . . . , S(AN, B). Since
h0 separates C1 and C2, by the multiplicative property of the similarity we have,
S(C1, C2) = S(C1, h0)S(h0, C2) = uσvT
∥u∥= ∥v∥= 1.
Let ¯β be the proportionality constant between u and S(C1, h0) such that u = ¯βS(C1, h0). Recall
that uA, uB in Eq. (8) are the entries in u that correspond to A and B, respectively. Partitioning
u into uA and uB and partitioning S(C1, h0) into S(A, h0) and S(B, h0) gives
uA = ¯βS(A, h0)
uB = ¯βS(B, h0).
It follows that
S(A, B) −uAαuT
B = S(A, B) −α¯β2S(A, h0)S(h0, B) = S(A, B) −βS(A, h0)S(h0, B),
where β = ¯β2α. We split S(A, B) into the submatrices S(A0, B), S(A1, B), ..., S(AN, B). Similarly,
17

## Page 18

we split S(A, h0) into the components S(A0, h0), S(A1, h0), ..., S(AN, h0). This gives
S(A, B) −βS(A, h0)S(h0, B) =


S(A0, B)
S(A1, B)
...
S(AN, B)

−β


S(A0, h0)
S(A1, h0)
...
S(AN, h0)

S(h0, B).
(23)
Let Ri = S(h0, hi)2. We show that the matrix S(Ai, h0)S(h0, B), which appears on the right side
of Eq. (23), is proportional to S(Ai, B) with the proportionality constant Ri.
RiS(Ai, B) = S(h0, hi)2S(Ai, hi)S(hi, B)
= S(Ai, hi)S(hi, h0) S(h0, hi)S(hi, B) = S(Ai, h0)S(h0, B).
(24)
Inserting (24) into (23) gives


S(A0, B)
S(A1, B)
...
S(AN, B)

−β


R0S(A0, B)
R1S(A1, B)
...
RNS(AN, B)

=


(1 −βR0)S(A0, B)
(1 −βR1)S(A1, B)
...
(1 −βRN)S(AN, B)

.
Thus, the score in Eq. (8) is equivalent to
d2(e) =
1
∥S(A, B)∥2
F
min
β
N
X
i=0
(1 −βRi)2∥S(Ai, B)∥2
F .
(25)
Since ∥S(A, B)∥2
F = PN
i=0 ∥S(Ai, B)∥2
F , we can rewrite Eq. (25) as follows,
d2(e) = min
β
PN
i=0(1 −βRi)2∥S(Ai, B)∥2
F
PN
i=0 ∥S(Ai, B)∥2
F
.
(26)
Next, the following lemma, proven in Appendix F, bounds the ratio of two sums.
Lemma 5.7. For two series of positive numbers ai, bi > 0 we have
P ai
P bi
≥
min
i̸=j;|i−j|≤2
ai + aj
bi + bj
.
Applying the lemma to Eq. (26) yields
d2(e) ≥
min
0≤i≤N−1;k∈{1,2} min
β
(1 −βRi)2∥S(Ai, B)∥2
F + (1 −βRi+k)2∥S(Ai+k, B)∥2
F
∥S(Ai, B)∥2
F + ∥S(Ai+k, B)∥2
F
.
(27)
Combining Eq. (27) and Lemma 5.6 gives
d2(e) ≥





(2δ2)log mδ6(1−ξ2)2
4mξ4
δ2 ≤0.5
δ8(1−ξ2)2
2mξ4
δ2 > 0.5,
which concludes the proof of Lemma 5.5, and with it, Step 1 in the proof of Theorem 5.4.
18

## Page 19

Step 2: A suﬃcient condition on the estimate ˆS
Lemma 5.5 shows that there is a gap
between the score of the correct placeholder edge and the scores of all other edges in T1. In the
following lemma we show that if ˆS is suﬃciently close to S the gap is preserved and STDR selects
the correct placeholder edge. For simplicity, we address only the case δ2 ≥0.5.
Lemma 5.8. Let D = min{∥S(A, B)∥F , ∥S(C1, C2)∥F }. If the similarity matrix estimate ˆS satisﬁes
∥S −ˆS∥F ≤1
2
 
2
D + 2.5
D2 + 1 + 10
√
2
D3
!−1
δ3(1 −ξ2)
√
2mξ2 ,
(28)
then STDR selects the correct placeholder edge.
In our proof, we use the following auxiliary lemma, proven in Appendix F.
Lemma 5.9. Let d(e), ˆd(e) be the exact and estimated score functions. If ∥S −ˆS∥F ≤D/2, then
|d(e) −ˆd(e)| ≤∥S −ˆS∥F
 
2
D + 2.5
D2 + 1 + 10
√
2
D3
!
.
Proof of Lemma 5.8. Suppose e∗∈T1 is the correct placeholder edge and e′ ̸= e∗is a diﬀerent edge
in T1. By Lemma 5.5
d(e′) ≥δ3(1 −ξ2)
√
2mξ2 ,
while for the correct edge d(e∗) = 0. It follows from the triangle inequality that if
|d(e) −ˆd(e)| ≤1
2
δ3(1 −ξ2)
√
2mξ2 ,
(29)
for all edges e, then ˆd(e∗) < ˆd(e′). Since δ ≤ξ < 1 and m ≥2,
1
2
 
2
D + 2.5
D2 + 1 + 10
√
2
D3
!−1
δ3(1 −ξ2)
√
2mξ2
≤D
2 .
Thus, if the estimate ˆS satisﬁes Eq. (28), then ∥S −ˆS∥≤D/2 and the condition for Lemma 5.9
holds. Combining the lemma with Eq. (29) concludes the proof.
Step 3: Finite sample guarantees
We are now ready to prove Theorem 5.4, which bounds the
number of samples required to compute, with high probability, a suﬃciently accurate estimate ˆS,
as determined in Lemma 5.8.
Proof of Theorem 5.4. The following concentration bound for ˆS was derived in Lemma 4.7 of [26],
Pr(∥ˆS −S∥F ≤t) ≥1 −2m2 exp

−2nt2
ℓ2m2

.
We note that in [26], this bound was presented for the spectral norm, but the proof holds for the
Frobenius norm as well. Suppose that Pr(∥ˆS −S∥F ≤t) > 1 −ε, Namely
n ≥ℓ2m2
2t2 log
2m2
ε

.
(30)
19

## Page 20

By Lemma 5.8, a suﬃcient condition for STDR to select the correct placeholder edge is
∥S −ˆS∥F ≤1
2
 
2
D + 2.5
D2 + 1 + 10
√
2
D3
!−1
δ3(1 −ξ2)
√
2mξ2 .
(31)
Setting t to the right hand side of Eq. (31) and substituting into Eq. (30), we have that if
n ≥ℓ2m2
2
22
 
2
D + 2.5
D2 + 1 + 10
√
2
D3
!2 
2mξ4
δ6(1 −ξ2)2

log
2m2
ε

= 8ℓ2m3
 
2
D + 2.5
D2 + 1 + 10
√
2
D3
!2 
ξ4
δ6(1 −ξ2)2

log
2m2
ε

then Eq. (31) holds with probability at least 1 −ε, and thus the merging step in STDR selects the
correct placeholder edge with high probability.
Remark 5.10. The guarantees in Theorems 5.1 and 5.4 are derived for a single partitioning and
merging step. Since the algorithm is recursive, additional splitting and merging steps depend on
submatrices of ˆS. If the bounds in Lemmas 5.2 and 5.8 are satisﬁed for the full matrix S, they
hold simultaneously for all submatrices of S as well. Thus, the number of samples required in
Theorems 5.1 and 5.4 is suﬃcient to guarantee with high probability the success of STDR for
multiple partitioning and merging steps.
5.3
Comparison of sample complexity
Combining Theorem 5.1 and Theorem 5.4, for a binary symmetric tree with a ﬁxed similarity
between adjacent nodes, the sample complexity of the partitioning and merging steps of STDR is
e
O

m3/D2
min + m2+4 log2( 1
δ )
(32)
where Dmin is the minimum value of D from Theorem 5.4 over all partitions of the tree.
We
compare this result to three other methods for full recovery of trees. For simplicity, we assume
that the similarity between all adjacent nodes is δ. Thus, the value of Dmin = e
O(m3/δτ 2), where
τ is the user given threshold. For a reasonalbe setting where τ = e
O(m3/δτ 2), the sample com-
plexity simpliﬁes to e
O(m2+4 log2( 1
δ )). For NJ, the sample complexity given in Section 3.3 of [4] is
e
O(exp(−4 mini,j ln(S(xi, xj))) or equivalently e
O
 δ−4diam(T )
, where diam(T ) is the diameter of
T . For a binary symmetric tree diam(T ) = 2 log2(m) and hence the complexity is O(m8 log2(1/δ)),
which is better than (32) for δ close to one, but worse for lower values of δ. However, the diameter
of the tree can be as large as m, in which case the sample complexity of NJ is exponential in m,
rather than polynomial as in (32).
For SNJ, if δ2 > 0.5 the sample complexity is e
O(m2) (by Theorem 4.3 in [26]). This is similar
to (32) for δ close to one, but improves upon (32) as δ decreases.
For the Dyadic Close method [15, Theorem 9], the sample complexity is e
O((1/δ)4h(T )), where
recall that h(T ) denotes the depth of a tree as in deﬁnition 7. For a binary symmetric tree h(T ) =
log2(m) in which case the complexity is O(m4 log2(1/δ)), which improves upon Eq. (32) by m2. For
highly imbalanced trees depth(T ) = O(1) in which case the sample complexity is logarithmic in m.
20

## Page 21

The improved sample complexity, however, comes at cost of a O(m5) computational complexity.
Thus, excluding the Dyadic Closure method, the sample complexity of STDR is similar to several
other distance-based methods with theoretical guarantees.
6
Simulation Results
We illustrate the performance of STDR in comparison to several other algorithms in a variety of
simulated settings. To this end we generated trees according to the coalescent model (6.1) and
the birth-death model (6.2), which are common in phylogenetics. In addition, we also considered
the challenging scenario of the caterpillar tree. In all experiments, the sequences were generated
according to the HKY substitution model [24] with transition-transversion ratio of 2, a typical value
in the human genome [31]. The mutation rate for the HKY model is speciﬁed for each simulation.
We considered the following reconstruction methods: (i) RAxML [52], a standard tool for max-
imum likelihood-based tree inference, (ii) neighbor joining (NJ), and (iii) spectral neighbor joining
(SNJ). Recall that STDR requires as input a subroutine alg for the reconstruction of the small
trees. Thus, for comparison, we applied STDR with each of the aforementioned algorithms as the
subroutine. We denote these three methods as (iv) STDR + RAxML, (v) STR + NJ and (vi)
STDR + SNJ. A second input to STDR is the threshold parameter τ, which sets an upper bound
for the size of the small trees. This parameter is speciﬁed in the description of each experiment. The
accuracy of the diﬀerent algorithms is measured by the normalized Robinson-Foulds (RF) distance,
deﬁned as the RF distance [17] between the reconstructed and reference tree divided by 2m −6.
Each experiment was repeated 5 times to obtain a mean and standard deviation of the performance
and runtime of each method.
In addition to the above experiments, we compare our merging procedure to TreeMerge [39].
The results for the caterpillar tree and the comparison to TreeMerge are shown in Appendix G.
Finally, for a symmetric binary tree, we demonstrate how changes in the threshold τ aﬀect the
results of STDR.
Implementation remarks
To improve the results of STDR, we computed two possible partitions
C1, C2: (i) A partition that corresponds to a threshold at 0 in the Fiedler vector, and (ii) a partition
that corresponds to the largest gap. In practice, the partition was chosen by method (i) or (ii), as
the one that minimizes the second singular value of S(C1, C2), see Lemma 3.1. To improve runtime,
we apply randomized methods for computing leading singular values and vectors, see [53, 22, 1].
6.1
Kingman’s coalescent model
We generated a random tree according to Kingman’s coalescent model [49] with m = 2000 terminal
nodes (See example in Fig 9). Figure 6 shows the accuracy (left panel) and the runtime (right
panel) of the diﬀerent methods as functions of the sequence length. The threshold parameter τ was
set to 128 for all experiments. Here, STDR+RAxML performs similarly to RAxML in accuracy
while achieving more than an order-of-magnitude reduction in runtime. Compared to NJ and SNJ,
STDR+NJ and STDR+SNJ show improvement in both accuracy and runtime.
21

## Page 22

2000
3000
4000
5000
6000
7000
8000
Number of samples (n)
0.2
0.3
0.4
0.5
0.6
normalized RF distance
2000
3000
4000
5000
6000
7000
8000
Number of samples (n)
102
103
104
runtime (s)
STDR+RAxML (128)
RAxML
STDR+NJ (128)
NJ
STDR+SNJ (128)
SNJ
Figure 6: Trees generated according to Kingman’s coalescent model with m = 2000 terminal nodes. The
mean and standard deviation of the normalized RF distance (left) between the reconstructed tree and the
input tree and of the runtime (right) are shown for each method over 5 independent runs.
800
1000
1200
1400
1600
Number of samples (n)
0.05
0.10
0.15
0.20
normalized RF distance
800
1000
1200
1400
1600
Number of samples (n)
102
103
104
runtime (s)
STDR+RAxML (256)
RAxML
STDR+NJ (256)
NJ
STDR+SNJ (256)
SNJ
Figure 7: A birth-death tree with m = 2048 terminal nodes.
The mean and standard deviation of the
normalized RF distance (left) between the reconstructed tree and the input tree and of the runtime (right)
are shown for each method over 5 independent runs.
6.2
Birth-death model
We generated random binary trees with m = 2048 terminal nodes according to the birth-death
model [33] .The STDR threshold was set to τ = 256 for all three methods. Figure 7 shows the
accuracy and runtime of the diﬀerent methods as a function of the sequence length n. Using STDR
with NJ clearly improves upon the performance of standard NJ both in terms of accuracy and
runtime. Compared to SNJ and RAxML, STDR+SNJ and STDR+RAxML show similar accuracy
but with signiﬁcantly faster runtimes
6.3
Eﬀect of threshold parameter
Our aim in this experiment was to test the impact of the threshold parameter τ on the performance
of STDR. To that end, we created a binary symmetric tree with m = 2048 terminal nodes and
similarity between all adjacent nodes equal to δ = 0.65. The number of samples was set to n = 1000.
We then reconstructed the tree via STDR with diﬀerent subroutines and a range of threshold values.
22

## Page 23

32
64
128
256
512
Threshold
0.0010
0.0015
0.0020
0.0025
0.0030
0.0035
normalized RF distance
32
64
128
256
512
Threshold
1000
2000
3000
runtime (s)
method
STDR+RAxML
STDR+SNJ
Figure 8: Eﬀect of minimal tree size τ on runtime and accuracy of SDTR. Various values of threshold τ
were chosen to test the performance of SDTR method in recovering a binary tree of size 2048 from sequences
of length 1000. SNJ, and RAxML were used as the sub method of SDTR.
Figure 8 shows the normalized RF distance between the recovered trees and the ground truth
tree as a function of the threshold. For both RAxML and SNJ, accuracy slightly improves for
higher values of the threshold. STDR + NJ is not shown in the plot because it is signiﬁcantly less
accurate in this setting. These results are in accordance with our analysis in Section 5, where we
show that the task of merging trees becomes challenging for small subsets of terminal nodes.
Acknowledgments
The authors would like to thank Junhyong Kim, Stefan Steinerberger and Ronald Coifman for use-
ful and insightful discussions. Y.K. and Y.A. acknowledge support by NIH grant R01GM131642,
UM1DA051410 and R61DA047037. Y.K. and B.N. acknowledge support by NIH grant R01GM135928.
Y.K. acknowledges support by NIH grant 2P50CA121974.
Appendix A
Example of Fiedler vector in a coalescent tree
We generated a tree with m = 512 nodes according to the coalescent model, see Figure 9a. The
transition matrices were set according to the HKY model [24]. We then generated a dataset of
nucleotide sequences of length n = 2, 000. Figure 9b shows the Fiedler vector of the similarity
graph estimated from the dataset. Partitioning the terminal nodes according to the sign pattern of
the Fiedler vector yields two clans.
Appendix B
Relation between the partitioning step and the
min-cut criterion
Let T be a binary tree and G be its similarity graph, as deﬁned in Section 4. The following lemma
shows that partitioning the terminal nodes according to the min-cut criterion yields two clans of
23

## Page 24

(a) Generated coalescent tree
0
100
200
300
400
500
−0.04
−0.02
0.00
0.02
0.04
0.06
(b) Fiedler vector of the coalescent tree
Figure 9: Coalescent tree example with 512 terminal nodes
T .
Lemma B.1. Let G be the similarity graph of a binary tree T . Let A∗and B∗be a partition of
the terminal nodes that minimizes the following min-cut criterion:
(A∗, B∗) ∈argmin
A,B
CutG(A, B) = argmin
A,B
X
i∈A,j∈B
S(xi, yj).
(33)
Then A∗and B∗are clans in T .
Proof. Let (x1, x2) be a pair of adjacent terminal nodes. Consider an arbitrary partition of the
terminal nodes into two non-empty subsets, denoted A and B. The two adjacent nodes (x1, x2)
can, respectively, be labeled (A, B), (A, A), (B, A) or (B, B).
We show that if A and B each
contains nodes besides x1 and x2, then assigning x1 and x2 to the same subset decreases the value
of the min-cut criterion.
Assume without loss of generality that x1 ∈A, x2 ∈B. The cut between A and B is equal to
Cut(A, B) ≡
X
x∈A,x′∈B
S(x, x′) = S(x1, x2) +
X
x′∈B\{x2}
S(x1, x′) +
X
x∈A\{x1}
S(x, x2) + S0,
where
S0 =
X
x∈A\{x1}
x′∈B\{x2}
S(x, x′)
does not depend on the assignment of x1 and x2. Let h be the unique node that is adjacent to both
x1 and x2. From the multiplicative property of the similarity, we have
Cut(A, B) = S(x1, x2) + S(x1, h)
X
x′∈B\{x2}
S(h, x′) + S(x2, h)
X
x∈A\{x1}
S(x, h) + S0.
Without loss of generality, assume that
X
x′∈B\{x2}
S(h, x′) ≥
X
x∈A\{x1}
S(x, h).
(34)
24

## Page 25

It follows that
Cut(A, B) ≥S(x1, h)
X
x∈A\{x1}
S(x, h) + S(x2, h)
X
x∈A\{x1}
S(x, h) + S0
(35)
=
X
x∈A\{x1}
S(x, x1) +
X
x∈A\{x1}
S(x, x2) +
X
x∈A\{x1}
x′∈B\{x2}
S(x, x′) =
X
x∈A\{x1}
x′∈B∪{x1}
S(x, x′).
Note that the right hand side of Eq. (35) equals the value of the cut of the same partition, but
with x1 moved from A to B. Thus, the min-cut partition {A∗, B∗} satisﬁes one of the following:
• x1 and x2 are in the same subset.
• One of A∗or B∗equals exactly to {x1} or {x2}.
Next, let C1 and C2 be two adjacent clans. Assume that the terminal nodes of each of the clans
are homogeneous (i.e., they all belong to the same subset, A or B). The same argument for a pair
of terminal nodes carries over to the case of two adjacent homogeneous clans, showing that the
minimal cut partition {A∗, B∗} satisﬁes one of the following:
• C1 and C2 are in the same subset.
• One of A∗or B∗equals exactly C1 or C2.
Let {A, B} be an arbitrary partition of the terminal nodes that does not correspond to two clans in
the tree. Since A and B are not clans, there must be at least two disjoint pairs C1, C2 and ˜C1, ˜C2
of homogeneous adjacent subsets, where the nodes in C1 are labeled by A and the nodes in C2 are
labeled by B. By our arguments Cut(A, B) can be reduced by either changing the labels of C1 to B
or C2 to A which implies that {A, B} is not the min-cut partition. Thus, for any min-cut partition
{A∗, B∗}, A∗and B∗are clans.
Appendix C
Supplementary proofs for Section 3
We present here the proofs of Lemmas 3.2 and Lemma 3.3 that are used in Section 3.
Proof of Lemma 3.2. Let C2 be the clan of all the terminal nodes of T that are not in C1. Consider
an edge e(hA, hB) in T1 that partitions C1 into A(e) and B(e). First, assume that e(hA, hB) is the
correct placeholder edge of T1. Then there exists a node h1 in the full tree T that is connected to
hA, hB and to the root node of C2. Removing the edge e(hA, h1) in T separates the subset A(e)
from the remaining nodes in T , which implies that A(e) is a clan in T . By the same argument,
B(e) is also a clan in T .
Conversely, assume that A(e), B(e) and C2 are disjoint clans that partition the terminal nodes
of T . Then, there exists a node h1 that connects to the roots of A(e), B(e) and T2. This proves that
the edge e(hA, hB) in T1 is the correct placeholder edge, since it is where the root h1 is inserted.
Proof of Lemma 3.3. Let C1 = A ∪B be the terminal nodes of the clan T1 and let h1 be its root.
We denote by C2 the terminal nodes in its adjacent clan. By the multiplicative property of the
similarity function,
S(C1, C2) = S(C1, h1)S(h1, C2).
25

## Page 26

Combining the above expression with Eq. (6) implies that the left singular vector u of S(C1, C2) is
proportional to the vector of similarities S(C1, h1). That is, ∃β ∈R such that S(C1, h1) = βu. Let
e be an edge in T1 that partitions the terminal nodes into A(e), B(e). The vector S(C1, h1) can be
similarly partitioned into S(A(e), h1) and S(B(e), h1) such that
S(A(e), h1) = βuA(e),
S(B(e), h1) = βuB(e).
(36)
We ﬁrst prove that if e is the correct placeholder edge of T1, then Eq. (7) holds. By Lemma 3.2, if
e is the correct placeholder edge then the root node h1 separates A(e) from B(e). By Eq. (36) and
the multiplicative property of the similarity measure, we have
S(A(e), B(e)) = S(A(e), h1)S(h1, B(e)) = uA(e)β2uT
B(e).
Setting α = β2 proves Eq. (7).
Next, we assume that Eq. (7) holds for some edge e and prove that e is the correct placeholder
edge. Consider the matrix S(A(e), B(e) ∪C2). Since h1 is the root of T1,
S(A(e), C2) = S(A(e), h1)S(h1, C2)
and
S(A(e), h1) = βuA(e)
we have
S(A(e), C2) = βuA(e)S(h1, C2).
Recall that by assumption S(A(e), B(e)) = uA(e)αuB(e). It follows that both matrices S(A(e), B(e))
and S(A(e), C2) are rank one with a left singular vector equal to uA(e). Thus, the concatenated
matrix S(A(e), B(e) ∪C2) is rank-one. By Lemma 3.1, this implies that A(e) is a clan of the tree
T . A similar argument shows that B(e) is also a clan in T . Since A(e) and B(e) are both clans in
T , it follows from Lemma 3.2 that e is the correct placeholder edge of T1.
Appendix D
Comparison to distance based tree partitioning
Let D ∈Rm×m be a matrix whose elements are the pairwise phylogenetic distances between all
terminal nodes. Given the exact distance matrix, it was shown in [20] that the terminal nodes of
a tree can be partitioned into two clans according to the sign pattern of the leading eigenvector of
the following matrix
(I −11T /m)D(I −11T /m).
Figure 10 shows the percentage of times the terminal nodes were correctly partitioned into clans
by applying our similarity based approached vs. the distance-based approach derived in [20]. We
generated 200 random trees according to Kingman’s coalescent model with m = 128 terminal nodes.
Figures 10a shows the ratio of times each method successfully partitioned the tree as a function of
the number of samples with a ﬁxed mutation rate between adjacent nodes of δ = 0.9. Similarly,
Figure 10b shows the performance of both methods as a function of δ with a ﬁxed number of samples
n = 100. The advantage of using the similarity matrix over the distance matrix is clear.
Appendix E
Proof of Lemma 4.4
We begin with several deﬁnitions and notations. We denote by G(v, w), T (v, w) the weight between
nodes v and w in a graph G and tree T , respectively. For a tree T , we denote by pathT (v, w) the
26

## Page 27

75
100
150
200
N
0.800
0.825
0.850
0.875
0.900
0.925
0.950
0.975
Correct  ratio
Method
Distance based
Similarity based
(a) Partitioning accuracy vs. number of samples.
0.84
0.86
0.88
0.9
0.92

0.75
0.80
0.85
0.90
0.95
1.00
Correct ratio
Method
Distance based
Similarity based
(b) Partitioning accuracy vs. mutation rate.
Figure 10: Comparison between distance based and similarity based spectral partitioning.
set of edges on the path between nodes v and w,
pathT (v, w) = {(˜v, ˜w)| ˜v and ˜w are adjacent nodes on the path between v and w}.
Next, we deﬁne the multiplicative weight between two nodes in a tree.
Deﬁnition 8. The multiplicative weight between v and w in a tree T is equal to,
αT (v, w) =
Y
(˜v, ˜
w)∈pathT (v,w)
T (˜v, ˜w).
(37)
For example, let T be a tree whose edge weights are given by the similarity in Eq. (4), then the
similarity between two terminal nodes x1, x2 is equal to the multiplicative weight αT (x1, x2). The
next deﬁnition concerns a graph with nodes that correspond to a subset of nodes in T , and weights
computed according to (37).
Deﬁnition 9 (Multiplicative subgraph). Let T be a tree with a set of nodes V . We say that a
graph G is a multiplicative subgraph with respect to T and a subset of nodes eV ⊂V if (i) the nodes
of G correspond to eV and (ii) the weight assigned to an edge connecting v, w in G is equal to the
multiplicative weight between v and w in T ,
G(v, w) = αT (v, w).
For convenience, we will sometimes say that G is a multiplicative subgraph of T without explic-
itly stating which nodes are included in G. By deﬁnition, the similarity graph G is a multiplicative
subgraph with respect to the terminal nodes of T . Note that we use v and w as nodes both in G
and in T interchangeably, since by deﬁnition every node in G corresponds to a node in T .
The proof of Lemma 4.4 is constructive. Given a tree T and its similarity graph G, we present
an iterative procedure to build a second tree ˜T , with the same topology as T , but with diﬀerent
weights such that
LG = L ˜T /R,
27

## Page 28

where R is the set of all internal nodes in T . Computing ˜T consists of iterative and simultaneous
updates of a graph and a tree: (i) a graph Gi with nodes that correspond to a subset of the nodes
in T . The initial graph G0 is set to G, with only the terminal nodes of T . (ii) A tree Ti, with the
same topology as T . The weights of the initial tree T0 are set such that T0 = T .
At each iteration i, we add one of the non-terminal nodes hi of T (that was not previously
added) to Gi, creating Gi+1.
The weights of the new graph Gi+1 are set such that the Schur
complement of its Laplacian matrix with respect to the added node hi is equal to the Laplacian of
the previous graph LGi.
LGi = LGi+1/hi.
(38)
The steps for computing Gi+1 given Gi and Ti are described in Algorithm 2. Next, we compute a
new tree Ti+1 with the same topology as Ti. The weights of Ti+1 are set such that Gi+1 becomes
a multiplicative subgraph with respect to Ti+1. The steps for computing Ti+1 are described in
Algorithm 3. At every iteration i, we maintain an active set of nodes which we denote by Ai.
When updating Gi, changes are only made to edges connecting two nodes in Ai ∪hi.
When
updating Ti, changes are only made to edges on the path between two nodes in the active set. The
initial active set A0 is equal to all terminal nodes of T .
In our proof, we use the following two auxiliary lemmas, that show the correctness of the
updating procedure of Gi and Ti. An implementation of Algorithms 2 and 3 is available on GitHub.
The ﬁrst lemma proves the correctness of Algorithm 2. The input to Algorithm 2 is the tree Ti,
a multiplicative subgraph Gi and an active set Ai, all of which were computed in the previous
iteration. The output of the algorithm is an updated graph Gi+1 that contains an additional node
hi. In addition, the algorithm updates the active set Ai and creates Ai+1.
Lemma E.1. The output of Algorithm 2 is a graph Gi+1 whose nodes include hi as well as all the
nodes in Gi such that
LGi+1/hi = LGi.
The next lemma concerns the updating procedure of Ti. The input to Algorithm 3 consists of
the new active set Ai+1, and the node hi added to Gi+1. Here, the only changes made are to edges
on the path between hi and the nodes in the active set Ai+1.
Lemma E.2. The tree Ti+1 built according to Algoithm 3 is such that Gi+1 becomes a multiplicative
subgraph of Ti+1.
Figure 11 shows two iterations of the aforementioned process for a tree T with four terminal
and two non-terminal nodes. For simplicity, all the weights of the tree T are set to 1/2.
Proof of Lemma 4.4. We initialize the updating process with a tree T and its similarity matrix
G = G0. By deﬁnition, G0 is a multiplicative subgraph of T , and therefore satisﬁes the condition
for Lemma E.1. The lemma guarantees that after the ﬁrst update, we obtain a graph G1 with a
Laplacian that satisﬁes,
LG0 = LG1/h0,
where h0 is the node added to G0 at the ﬁrst iteration.
Lemma E.2 guarantees that G1 is a
multiplicative subgraph of T1. Thus, we can re-apply Algorithm 2 with the pair G1, T1. Thus, at
each iteration i, we obtain a graph Gi+1 that satisﬁes,
LGi = LGi+1/hi.
(42)
28

## Page 29

Algorithm 2 Updating Gi
Input: Ti
- a tree graph
Gi
- a multiplicative subgraph of Ti
Ai
- active set of nodes
Output: Gi+1
- updated graph such that LGi = LGi+1/hi
Ai+1
- updated active set
hi
- the node added to Gi
vi,1, vi,2
- nodes removed form the active set
1: Initialize Gi+1 = Gi and Ai+1 = Ai.
2: Choose a node hi in Ti that is not in Gi and is adjacent to at least two nodes vi,1, vi,2 in the
active set Ai. Add hi to Gi+1.
3: Remove edges between the nodes vi,1, vi,2 and the rest of the active set Ai.
4: The weight between the new node hi and a node x in the active set is computed by
Gi+1(hi, x) = dαTi(hi, x),
(39)
where
d =
X
x′∈Ai
αTi(hi, x′).
(40)
5: The weights between two nodes x, y in the active set (except vi,1, vi,2) are updated by
Gi+1(x, y) = Gi(x, y) −αTi(hi, x)αTi(hi, y).
(41)
6: Remove the nodes vi,1 and vi,2 from the active set Ai+1, and add hi.
7: return Gi+1, Ai+1, hi,vi,1,and vi,2.
Repeating the updating process for all l non-terminal nodes of T yields the graph Gl, which by con-
struction has the same topology as T . In addition, due to the transitivity of the Schur’s complement
operation, Eq. (42) implies that
LTl/R = LGl/R = LGl/{h0,...hl−1} = LGl−1/{h0,...hl−2} = . . . = LG1/h0 = LG0 = LG.
Thus, Tl is a tree with the same topology as T , but with diﬀerent weights such that LTl/R = LG,
which proves the lemma.
Proof of Lemma E.1. Assume, for simplicity of notation, that the jth row/column of LG is the
row/column that correspond to hj for any j such that
LG(i, j) = −G(hi, hj)
∀hi, hj ∈G with i ̸= j.
We denote by mj the j-th column of LGi+1 after removing the i-th entry, and by 1 the all one
vector. Since hi is a single node, the Schur complement LGi+1/hi deﬁned in (6) can be simpliﬁed to
LGi+1/hi(j, k) = LGi+1(j, k) −(1T mj)(1T mk)
P
l̸=i 1T ml
.
(43)
29

## Page 30

Algorithm 3 Updating Ti
Input: Ti
- a tree graph
Ai+1
- the active set
hi
- the node last added to Gi+1
vi,1, vi,2
- nodes that where removed from the active set in the last update
Output: Ti+1
- a tree with weights computed such
that Gi+1 is a multiplicative subgraph of Ti+1
1: Set Ti+1(hi, vi,1) = dTi(hi, vi,1) and Ti+1(hi, vi,2) = dTi(hi, vi,2)
2: For node x /∈{vi,1, vi,2} adjacent to hi, set
Ti+1(hi, x) =
dTi(hi, x)
p
1 −αTi(x, hi)2 ,
where d is given by Eq. (40).
3: For two adjacent nodes x, y ∈T where y is a node in the active set Ai+1 and x is other path
between y and hi, set
Ti+1(x, y) = Ti(x, y)
p
1 −αTi(x, hi)2
4: For two adjacent nodes x, y ∈T that are not in the active set. If Ti(x, y) is on the path between
a node in the active set and hi, where x is closer to hi, set
Ti+1(x, y) = Ti(x, y)
p
1 −αTi(x, hi)2
p
1 −αTi(y, hi)2
5: return Ti+1
For a Laplacian matrix, the sum over any row is equal to zero. Since mj is equal to the row of
LGi+1 after removing the i-th entry we have that 1T mj = −LGi+1(i, j). We rewrite Eq. (43) as,
LGi+1/hi(j, k) = LGi+1(j, k) + LGi+1(j, i)LGi+1(k, i)
P
l̸=i LGi+1(i, l)
.
(44)
The only edges changed between Gi and Gi+1 are edges between nodes in the active set Ai. Thus,
if either hk or hj are not in the active set then LGi+1(j, k) = LGi(j, k). In addition, by step 4 of
Algorithm 2, the added node hi is only connected to nodes in the active set Ai. Thus, if either
node hk or hj are not part of Ai we have LGi+1(j, i)LGi+1(k, i) = 0. It follows that in this case
LGi+1/hi(j, k) = LGi(j, k) as required.
Next, we assume that both hj and hk are part of the active set Ai. Eqs. (39) and (41) give
LGi+1(j, k) = LGi(j, k) + αTi(hi, hj)αTi(hi, hk),
LGi+1(k, i) = −dαTi(hi, hk).
(45)
By step 4 of Algorithm 2, hi is only connected to nodes in the active set Ai. Inserting Eq. (45) to
Eq. (44) gives
LGi+1/hi(j, k) = LGi(j, k) + αTi(hi, hj)αTi(hi, hk) −d2αTi(hi, hj)αTi(hi, hk)
P
x∈Ai dαTi(hi, x)
,
(46)
30

## Page 31

The denominator in the last term on the r.h.s of Eq. (46) is equal to d2 and hence,
LGi+1/hi(j, k) = LGi(j, k) + αTi(hi, hj)αTi(hi, hk) −d2αTi(hi, hj)αTi(hi, hk) 1
d2 = LGi(j, k).
We conclude that for any element j, k we have LGi+1/hi(j, k) = LGi(j, k).
Proof of Lemma E.2. Here, our task is to prove that the weight assigned to any edge Gi+1(x, y) is
equal to the multiplicative path αTi+1(x, y). We address three cases: (i) the node x is in the active
set Ai+1 and y is equal to the node hi added to the graph in iteration i. (ii) Both x and y are in
Ai+1, and are not equal to hi, and (iii) x = hi and y is either vi,1 or vi,2. For a pair of nodes (x, y)
that is not in (i) −(iii) the edges in Gi and Ti were not changed in the updating steps.
For case (i) we assume that x is in Ai+1 and y = hi and hence by Eq. (39) in Algorithm 2
Gi+1(x, hi) = dαTi(x, hi).
We denote the nodes on the path between x and hi in Ti by
path(hi, x) = {z1 = hi, z2, . . . , zK = x}.
The edge between hi and z2 is updated according to step 2 of Algorithm 3. The edge between
zK−1 and zK is updated by step 3. The remaining edges are updated by step 4. The multiplicative
weight αTi+1(x, hi) in the updated tree Ti+1 according to Algorithm 3 is equal to
αTi+1(x, hi) =
K−1
Y
j=1
Ti+1(zj, zj+1)
=
dTi(hi, z2)
p
1 −αTi(z2, hi)2 ×
K−2
Y
j=2
Ti(zj, zj+1)
p
1 −αTi(zj, hi)2
p
1 −αTi(zj+1, hi)2
p
1 −αTi(zK−1, hi)2Ti(zK−1, x)
= d
K−1
Y
j=1
Ti(zj, zj+1) = dαTi(x, hi).
(47)
Thus, the weight Gi+1(x, hi) = αTi+1(x, hi) for any x in the active set.
In case (ii) x, y are two nodes in the active set not equal to hi.
According to Eq.
(41) in
Algorithm 2
Gi+1(x, y) = Gi(x, y) −αTi(hi, x)αTi(hi, y).
Denote by u the unique node that connects between the nodes x, y and hi. Then,
αTi(hi, x)αTi(hi, y) = αTi(hi, u)2αTi(u, x)αTi(u, y) = αTi(hi, u)2αTi(x, y).
(48)
By assumption on the input to Alg. 2 of the previous iteration, the graph Gi is a multiplicative
subgraph of Ti and hence Gi(x, y) = αTi(x, y). Thus, Eqs. (41) and (48) imply
Gi+1(x, y) = Gi(x, y)−αTi(hi, u)2αTi(x, y) = Gi(x, y)−αTi(hi, u)2Gi(x, y) = Gi(x, y)(1−αTi(hi, u)2).
31

## Page 32

Next, we show that Gi+1(x, y) is equal to the multiplicative weight αTi+1(x, y). Let z1 = x, . . . , zκ =
u, . . . , zK = y be the nodes on the path between x and y. By steps 2 and 3 in Algorithm 3, the
multiplicative weight αTi+1(x, y) is equal to
αTi+1(x, y) =
κ−1
Y
j=1
Ti+1(zj, zj+1)
K−1
Y
j=κ
Ti+1(zj, zj+1)
= Ti(x, z2)
p
1 −αTi(z2, hi)2
κ−1
Y
j=2
Ti(zj, zz+1)
p
1 −αTi(zj+1, hi)2
p
1 −αTi(zj, hi)2
× Ti(y, zK−1)
p
1 −αTi(zK−1, hi)2
K−2
Y
j=κ
Ti(zj, zz+1)
p
1 −αTi(zj, hi)2
p
1 −αTi(zj+1, hi)2 .
(49)
Note that
Ti(x, z2)
p
1 −αTi(z2, hi)2
κ−1
Y
j=2
Ti(zj, zz+1)
p
1 −αTi(zj+1, hi)2
p
1 −αTi(zj, hi)2
=
p
1 −αTi(zκ, hi)2
κ−1
Y
j=1
Ti(zj, zz+1)
and
Ti(y, zK−1)
p
1 −αTi(zK−1, hi)2
K−2
Y
j=κ
Ti(zj, zz+1)
p
1 −αTi(zj, hi)2
p
1 −αTi(zj+1, hi)2
=
p
1 −αTi(zκ, hi)2
K−1
Y
j=κ
Ti(zj, zz+1)
and thus,
αTi+1(x, y) =
K−1
Y
j=1
Ti(zj, zz+1)(1 −αTi(zκ, hi)2) = Gi+1(x, y).
Lastly, we consider case (iii), where x = hi and y = vi,1 or y = vi,2. Recall that vi,1, vi,2 are
adjacent to hi in T and were removed from the active set. By step 4 of Algorithm 2 and step 1
of Algorithm 3 the edge Gi(x, y) and its corresponding edge Ti(x, y) have both been updated such
that Ti+1(x, y) = Gi+1(x, y) = dTi(x, y).
Appendix F
Auxiliary Lemmas for Section 5
Proof of Lemma 5.3. We begin by characterizing all the eigenvectors of L ∈Rm×m. For any non-
terminal node h in the binary symmetric tree T , we denote the set of descendent terminal nodes
to the “left” of h by A, the set of descendant terminal nodes to the “right” of h by B, and the rest
of the terminal nodes by C. Let vh ∈Rm be a vector with
(vh)i =





1
i ∈A
−1
i ∈B
0
i ∈C.
32

## Page 33

We show that for any choice of non-terminal node h, vh is an eigenvector of L. Since there are
m −1 non-terminal nodes, this set of eigenvectors, together with the vector of all-ones, forms the
full set of all eigenvectors of L.
First, we show that vh is an eigenvector of the similarity matrix S, and compute the correspond-
ing eigenvalue. For i ∈A,
(Svh)i =
X
j∈A
S(i, j) −
X
k∈B
S(i, k).
Due to the symmetry of the tree T , every terminal node has a similarity of δ2 to one other terminal
node, δ4 to two other terminal nodes, etc. Thus,
X
j∈A
S(i, j) = 1 + δ2 + 2δ4 + . . . , + . . . , |A|δ2 log2 |A| = δ2
1 −(2δ2)log2 |A|
1 −2δ2

+ 1.
The similarity between a node i ∈A and all nodes k ∈B is equal to δ2(log |A|+1). Thus,
X
j∈A
S(i, j) −
X
k∈B
S(i, k) = δ2
1 −(2δ2)log2 |A|
1 −2δ2

+ 1 −|A|δ2(log |A|+1)
= 1 + δ2
1 −(2δ2)log |A|(2 −2δ2)
1 −2δ2

.
(50)
The same result with a negative sign holds for i ∈B. If i ∈C then by symmetry (Svh)i = 0. Thus
vh is an eigenvector of S with eigenvalue equal to the right side of (50). The sum of every row in
S is equal to,
di =
X
j
S(i, j) = 1 + δ2 + 2δ4 + . . . + 2log2 mδ2 log2 m = δ2
1 −(2δ2)log2 m
1 −2δ2

+ 1.
(51)
Let D be the scalar matrix with diagonal elements equal to Eq. (51). Combining Eq. (51) and Eq.
(50), we get that vh is an eigenvector of L = D −S with eigenvalue:
λ(h) = δ2
(2δ2)log2 |A|(2 −2δ2) −(2δ2)log2 m
1 −2δ2

.
(52)
For any Laplacian matrix 0 is an eigenvalue that correspond to the vector of all-ones. Since the
eigenvalue e(h) decreases as |A| grows, the two smallest non-zero eigenvalues correspond to |A| =
m/2 and |A| = m/4. The three smallest eigenvalues are thus equal to,
λ1 = 0,
λ2 = m2 log2(δ)+1,
λ3 = m2 log2(δ)+1
1
2 +
1
2δ2

.
In the following proof, we use similar notations as in the proof of Lemma 5.5.
Proof of Lemma 5.6 . For simplicity, let x = ∥S(Ai, B)∥2
F and y = ∥S(Ai+k, B)∥2
F . To compute
the numerator of Eq. (22), we set the partial derivative w.r.t. β to 0, which gives
β∗= argmin
β

(1 −βRi)2x + (1 −βRi+k)2y

= Rix + Ri+ky
R2
i x + R2
i+ky .
33

## Page 34

Plugging β∗back into the numerator of Eq. (22) gives
min
β

(1 −βRi)2x + (1 −βRi+k)2y

= xy(Ri −Ri+k)2
R2
i x + R2
i+ky .
Observe that Ri+k = RiS(hi, hi+k)2. Thus, the above expression further simpliﬁes to
xy(Ri −Ri+k)2
R2
i x + R2
i+ky
= xyR2
i (1 −S(hi, hi+k)2)2
R2
i (x + S(hi, hi+k)4y)
= xy(1 −S(hi, hi+k)2)2
x + S(hi, hi+k)4y
.
Since ∥S(Ai, B)∥2
F + ∥S(Ai+k, B)∥2
F = x + y, the LHS of (22) is equal to
xy(1 −S(hi, hi+k)2)2
(x + y)(x + S(hi, hi+k)4y).
(53)
Recall from Eqs. (2) and (3) that for any 1 ≤i ≤N −1, S(hi, hi+k) < ξ < 1. It follows that
xy(1 −S(hi, hi+k)2)2
(x + y)(x + S(hi, hi+k)4y) ≥xy(1 −ξ2)2
(x + y)2
≥
xy(1 −ξ2)2
(2 max(x, y))2 = (1 −ξ2)2 min(x, y)
4 max(x, y)
.
(54)
Next, we simplify the term
min(x,y)
max(x,y) in Eq. (54). Note that hi+k separates Ai and Ai+k from B,
see ilustration in Figure 5. Thus, we can rewrite min(x, y) as
min(x, y) = min(∥S(Ai, B)∥2
F , ∥S(Ai+k, B)∥2
F )
= min(∥S(Ai, hi+k)S(hi+k, B)∥2
F , ∥S(Ai+k, hi+k)S(hi+k, B)∥2
F )
= min(∥S(Ai, hi+k)∥2∥S(hi+k, B)∥2, ∥S(Ai+k, hi+k)∥2∥S(hi+k, B)∥2
F )
= min(∥S(Ai, hi+k)∥2, ∥S(Ai+k, hi+k)∥2) · ∥S(hi+k, B)∥2.
Similarly, max(x, y) = max(∥S(Ai, hi+k)∥2, ∥S(Ai+k, hi+k)∥2) · ∥S(hi+k, B)∥2. Thus,
min(x, y)
max(x, y) = min(∥S(Ai, hi+k)∥2, ∥S(Ai+k, hi+k)∥2)
max(∥S(Ai, hi+k)∥2, ∥S(Ak+1, hi+k)∥2).
Next, we provide lower and upper bounds on the terms ∥S(Ai, hi+k)∥2 and ∥S(Ai+1, hi+k)∥2. By
Eq. (2), the similarity between the nodes in Ai, Ai+k and hi+k is bounded by ξ. It follows that
max(∥S(Ai, hi+k)∥2, ∥S(Ai+k, hi+k)∥2) ≤max(|Ai|, |Ai+k|)ξ2 ≤mξ2.
(55)
For a lower bound, we apply [26, Lemma 4.5]. Given the terminal nodes of a clan A, and the root
of a clan h, the lemma bounds the norm of S(A, h) by,
∥S(A, h)∥2
F ≥
(
(2δ2)log |A|
δ2 ≤0.5
2δ2
δ2 > 0.5 ≥
(
(2δ2)log m
δ2 ≤0.5
2δ2
δ2 > 0.5.
There are k + 1 edges between the root of Ai and hi+k, and one edge between the root of Ai+k and
hi+k. Thus,
min(∥S(Ai, hi+k)∥2, ∥S(Ai+k, hi+k)∥2) ≥
(
(2δ2)log mδ2(k+1)
δ2 ≤0.5
2δ(2k+2)
δ2 > 0.5.
(56)
Plugging Eqs. (55), (56) into (54) concludes the proof.
34

## Page 35

Proof of Lemma 5.7 . The lemma is a small variation over the known lower bound for ratio of
sums,
P
i ai
P
i bi ≥mini ai
bi . For an even number of elements, we can merge non overlapping pairs of
consecutive elements such that ˜ai = a2i + a2i+1 and ˜bi = b2i + b2i+1. Applying the standard bound
for ratio of sums for ˜ai and ˜bi gives,
P
i ˜ai
P
i ˜bi
≥min
i
˜ai
˜bi
= min
i
a2i + a2i+1
b2i + b2i+1
≥
min
i̸=j;|i−j|≤2
ai + aj
bi + bj
.
For an odd number of elements, we can merge the ﬁrst three elements i = 0, 1, 2. The rest will be
merged into consecutive pairs.
P
i ai
P
i bi
≥min
(
a0 + a1 + a2
b0 + b1 + b2
,
P
i≥2(a2i + a2i+1)
P
i≥2(b2i + b2i+1)
)
The ratio for elements i = 0, 1, 2 can be bounded by the minimum ratio over all pairs i, j ∈{0, 1, 2}.
Thus,
P
i ai
P
i bi
≥min
(
min
i̸=j∈{0,1,2}
ai + aj
bi + bj
,
P
i≥2(a2i + a2i+1)
P
i≥2(b2i + b2i+1)
)
≥
min
i̸=j;|i−j|≤2
ai + aj
bi + bj
Lemma F.1. Let X, X′ ∈Rm×n and let y, y′ > 0. Assume that ∥X′∥F ≤y′, then
X
y −X′
y′

F ≤1
y (∥X −ˆX∥F + |y −ˆy|).
(57)
Proof. By deﬁnition,
X
y −
ˆX
ˆy

F =
Xy′ −X′y
yy′

F =
y′(X −X′)
yy′
+ X′(y′ −y)
yy′

F
By the triangle inequality
X
y −
ˆX
ˆy

F ≤1
y ∥X −X′∥F + |y′ −y|
y
· ∥X′∥F
y′
Since ∥X′∥F ≤y′ the lemma follows.
Lemma F.2. Let X and Y be two matrices and let ˆX and ˆY be their corresponding noisy estimates.
Then,
∥X −Y ∥F −∥ˆX −ˆY ∥F
 ≤∥X −ˆX∥F + ∥Y −ˆY ∥F .
Proof. Assume that ∥X −Y ∥F ≥∥ˆX −ˆY ∥F . In this case
∥X−Y ∥F −∥ˆX−ˆY ∥F
 = ∥X−Y ∥F −∥ˆX−ˆY ∥F ≤∥X−Y −ˆX+ ˆY ∥F ≤∥X−ˆX∥F +∥Y −ˆY ∥F .
Alternatively, if ∥X −Y ∥F ≤∥ˆX −ˆY ∥F we have
∥X−Y ∥F −∥ˆX−ˆY ∥F
 = ∥ˆX−ˆY ∥F −∥X−Y ∥F ≤∥ˆX−ˆY −X+Y ∥F ≤∥ˆX−X∥F +∥ˆY −Y ∥F .
35

## Page 36

Lemma F.3. Let X ∈Rn1×n2, Y ∈Rn2×n3, Z ∈Rn3×n4 be three matrices and let ˆX, ˆY , ˆZ be there
corresponding estimates. Then
∥XY Z −ˆX ˆY ˆZ∥F ≤∥X∥F ∥Y ∥F ∥Z −ˆZ∥F + ∥ˆZ∥F ∥Y ∥F ∥X −ˆX∥F + ∥ˆZ∥F ∥ˆX∥F ∥Y −ˆY ∥F
Proof.
∥XY Z−ˆX ˆY ˆZ∥F = ∥XY Z−XY ˆZ+XY ˆZ−ˆX ˆY ˆZ∥F ≤∥X∥F ∥Y ∥F ∥Z−ˆZ∥F +∥ˆZ∥F ∥XY −ˆX ˆY ∥F
(58)
Focusing on ∥XY −ˆX ˆY ∥F we have that
∥XY −ˆX ˆY ∥F = ∥XY −ˆXY + ˆXY −ˆX ˆY ∥F ≤∥X −ˆX∥F ∥Y ∥F + ∥ˆX∥∥Y −ˆY ∥F
Combining the two bounds gives,
∥XY Z −ˆX ˆY ˆZ∥F ≤∥X∥F ∥Y ∥F ∥Z −ˆZ∥F + ∥ˆZ∥F ∥Y ∥F ∥X −ˆX∥F + ∥ˆZ∥F ∥ˆX∥F ∥Y −ˆY ∥F
Lemma F.4. Let S denote a rank one matrix and ˆS its noisy estimate. We denote by u, ˆu their
respective leading left singular vectors. If ∥S −ˆS∥F ≤0.5∥S∥F then
∥uuT −ˆuˆuT ∥2
F ≤50∥S −ˆS∥2
F
∥S∥2
F
.
Proof.
∥uuT −ˆuˆuT ∥2
F =
X
ij
(uuT −ˆuˆuT )2
ij =
X
ij
(uuT )2
ij +
X
ij
(ˆuˆuT )2
ij −2
X
ij
(uuT )ij(ˆuˆuT )ij
= ∥u∥4 + ∥ˆu∥4 −2
X
i
uiˆui
X
j
uj ˆuj = 2(1 −(uT ˆu)2) = 2 sin2(u, ˆu).
(59)
We apply a variant of the Davis-Kahan theorem for non square matrices [61, Theorem 3]. The
perturbation of the leading singular vector is bounded by
sin(u, ˆu) ≤2(2σ1(S) + ∥S −ˆS∥)∥S −ˆS∥
σ2
1(S) −σ2
2(S)
,
where σ1(S) and σ2(S) are the two leading singular values of S. Since S is rank one, σ1(S) =
∥S∥= ∥S∥F and σ2(S) = 0. In addition, we assumed that ∥S −ˆS∥F ≤0.5∥S∥F and hence
sin(u, ˆu) ≤5∥S∥F ∥S −ˆS∥F
∥S∥2
F
= 5∥S −ˆS∥F
∥S∥F
.
(60)
Combining Eqs. (59), (60) concludes the proof.
Proof of Lemma 5.9 . Let d(e) denote the score of the edge e computed by the exact similarity
matrix S as deﬁned in (8). We denote by ˆd(e) the score computed by the noisy estimate of the
similarity ˆS. The diﬀerence between d(e) and ˆd(e) is equal to
|d(e) −ˆd(e)| =

∥S(A, B) −α∗uAuT
B∥F
∥S(A, B)∥F
−∥ˆS(A, B) −β∗ˆuAˆuT
B∥F
∥ˆS(A, B)∥F
 ,
(61)
36

## Page 37

where,
α∗= argmin
α
∥S(A, B) −αuAuT
B∥F
β∗= argmin
β
∥ˆS(A, B) −βˆuAˆuT
B∥F .
We apply Lemma F.1 with
X = ∥S(A, B) −α∗uAuT
B∥F ,
y = ∥S(A, B)∥F ,
ˆX = ∥ˆS(A, B) −β∗ˆuAˆuT
B∥F ,
ˆy = ∥ˆS(A, B)∥F ,
where we note that here X and ˆX are scalars. Lemma F.1 requires that 0 < | ˆX| ≤ˆy, which holds
trivially. Applying Lemma F.1 to (61) yields,
|d(e) −ˆd(e)| ≤
1
∥S(A, B)∥F
∥S(A, B) −α∗uAuT
B∥F −∥ˆS(A, B) −β∗ˆuAˆuT
B∥F

+
∥S(A, B)∥F −∥ˆS(A, B)∥F


.
(62)
Next, setting X = S(A, B), ˆX = ˆS(A, B), Y = α∗uAuT
B and ˆY = β∗ˆuAˆuT
B, by Lemma F.2,
|d(e) −ˆd(e)| ≤
1
∥S(A, B)∥F

∥S(A, B) −ˆS(A, B)∥F + ∥α∗uAuT
B −β∗ˆuAˆuT
B∥F
+
∥S(A, B)∥F −∥ˆS(A, B)∥F


≤1
D

2∥S(A, B) −ˆS(A, B)∥F + ∥α∗uAuT
B −β∗ˆuAˆuT
B∥F

.
(63)
where the second inequality is due to the reverse triangle inequality and the deﬁnition of D.
We focus on the term ∥α∗uAuT
B −β∗ˆuAˆuT
B∥F in Eq. (63). The values of α∗, β∗are obtained via
least square between the elements of S(A, B), ˆS(A, B) and uAuT
B, ˆuAˆuT
B, respectively. For α∗, the
least squares solution is
α∗=
1
∥S(A, B)∥2
F
X
i,j
S(A, B)ij(uAuT
B)ij =
1
∥S(A, B)∥2
F
uT
AS(A, B)uB,
(64)
where a similar expression holds for β∗. Multiplying α∗and β∗by uAuT
B and ˆuAˆuT
B gives,
α∗uAuB −β∗ˆuAˆuB =
1
∥S(A, B)∥2
F
uAuT
AS(A, B)uBuT
B −
1
∥ˆS(A, B)∥2
F
ˆuAˆuT
A ˆS(A, B)ˆuBˆuT
B.
(65)
Next, we apply Lemma F.1 with X = uAuT
AS(A, B)uBuT
B, y = ∥S(A, B)∥2
F , ˆX = ˆuAˆuT
A ˆS(A, B)ˆuBˆuT
B
and ˆy = ∥ˆS(A, B)∥2
F . The condition for Lemma F.1 is that ∥ˆX∥F ≤ˆy, which holds since
∥ˆX∥F = ∥ˆuAˆuT
A ˆS(A, B)ˆuBˆuT
B∥F ≤∥ˆuAˆuT
A∥F ∥ˆS(A, B)∥F ∥ˆuBˆuT
B∥F ≤∥ˆS(A, B)∥F = ˆy.
Applying Lemma F.1 to (65) gives
∥α∗uAuB −β∗ˆuAˆuB∥F ≤
1
∥S(A, B)∥2
F

∥uAuT
AS(A, B)uBuT
B −ˆuAˆuT
A ˆS(A, B)ˆuBˆuT
B∥F .
+
∥S(A, B)∥2
F −∥ˆS(A, B)∥2
F


.
(66)
37

## Page 38

Denote
ε(A, B) = ˆS(A, B) −S(A, B)
ε(C1, C2) = ˆS(C1, C2) −S(C1, C2)
εA =ˆuAˆuT
A −uAuT
A
εB = ˆuBˆuT
B −uBuT
B.
Equipped with the above notations, we bound the ﬁrst term in the numerator of Eq. (66) using
Lemma F.3 where X = uAuT
A, Y = S(A, B), and Z = uBuT
B,
∥uAuT
AS(A, B)uBuT
B −ˆuAˆuT
A ˆS(A, B)ˆuBˆuT
B∥F
≤∥uAuT
A∥F ∥S(A, B)∥F ∥εB∥F + ∥ˆuBˆuT
B∥F ∥S(A, B)∥F ∥εA∥F + ∥ˆuBˆuT
B∥F ∥ˆuAˆuT
A∥F ∥ε(A, B)∥F
Since ∥uAuT
A∥F , ∥ˆuBˆuT
B∥F ≤1 we get,
∥uAuT
AS(A, B)uBuT
B −ˆuAˆuT
A ˆS(A, B)ˆuBˆuT
B∥F ≤∥S(A, B)∥F (∥εA∥F + ∥εB∥F ) + ∥ε(A, B)∥F . (67)
The matrices εA, εB are submatrices of ˆuˆuT −uuT and hence ∥εA∥F , ∥εB∥F ≤∥ˆuˆuT −uuT ∥F .
Applying Lemma F.4 gives
∥εA∥F + ∥εB∥F ≤2∥uuT −ˆuˆuT ∥F ≤10
√
2∥ε(C1, C2)∥F
∥S(C1, C2)∥F
≤10
√
2∥ε(C1, C2)∥F
D
.
(68)
Combining Eqs. (63), (66),(67) and (68) yields
|d(e) −ˆd(e)| ≤1
D
 
2∥ε(A, B)∥F +
1
∥S(A, B)∥2
F
 
∥S(A, B)∥2
F −∥ˆS(A, B)∥2
F
 + ∥ε(A, B)∥F + 10
√
2∥S(A, B)∥F ∥ε(C1, C2)∥F
D
!!
.
(69)
We have that
∥S(A, B)∥2
F −∥ˆS(A, B)∥2
F
 =
∥S(A, B)∥F −∥ˆS(A, B)∥F
(∥S(A, B)∥F + ∥ˆS(A, B)∥F )
≤2.5∥ε(A, B)∥F ∥S(A, B)∥F ,
(70)
where the inequality is due to the reverse triangle inequality and our assumption ∥ε(A, B)∥F ≤
0.5∥S(A, B)∥F which implies ∥ˆS(A, B)∥F ≤1.5∥S(A, B)∥F . Combining (69) and (70), we get
|d(e) −ˆd(e)| ≤1
D

2∥ε(A, B)∥F +
1
∥S(A, B)∥2
F
×
 
∥ε(A, B)∥F (2.5∥S(A, B)∥F + 1) + 10
√
2∥ε(C1, C2)∥F ∥S(A, B)∥F
D
!!
≤∥ε(A, B)∥F
 2
D + 2.5
D2 + 1
D3

+ ∥ε(C1, C2)∥F
10
√
2
D3
≤∥S −ˆS∥F
 
2
D + 2.5
D2 + 1 + 10
√
2
D3
!
,
which concludes the proof.
38

## Page 39

Appendix G
Additional Simulation Results
G.1
Caterpillar tree
We generated a caterpillar tree with m = 512 terminal nodes, where the non-terminal nodes form a
path graph. The similarity between each pair of adjacent nodes was set to δ = 0.81. As in Section
6, we compare NJ, SNJ and RAxML, with STDR where the aforementioned methods are used as
subroutines. The STDR threshold is set to τ = 64 for all three STDR variants. Figure 12 shows
the normalized RF distance (left) and runtime (right) of the diﬀerent methods as functions of the
sequence length n. Here, all three methods are signiﬁcantly improved when combined with STDR
in both runtime and accuracy.
G.2
Comparison to TreeMerge
We generated random trees with 2000 terminal nodes according to the coalescent model. The trees
were recursively partitioned by STDR with a threshold of τ = 128. The structure of the diﬀerent
partitions was recovered by RAxML. We compared STDR’s merging criteria with TreeMerge [39]
for various sequence lengths. The results are shown in Figure 13. The merging process of STDR
achieved better accuracy than TreeMerge, with a signiﬁcantly reduced runtime.
References
[1] Yariv Aizenbud and Amir Averbuch. Matrix decompositions using sub-gaussian random ma-
trices. Information and Inference: A Journal of the IMA, 8(3):445–469, 2019.
[2] Elizabeth S Allman and John A Rhodes. Molecular phylogenetics from an algebraic viewpoint.
Statistica Sinica, 17(4):1299–1316, 2007.
[3] Anima Anandkumar, Daniel J Hsu, Furong Huang, and Sham M Kakade. Learning mixtures
of tree graphical models. In Advances in Neural Information Processing Systems, pages 1052–
1060, 2012.
[4] Kevin Atteson. The performance of neighbor-joining methods of phylogenetic reconstruction.
Algorithmica, 25(2-3):251–278, 1999.
[5] Sivaraman Balakrishnan, Min Xu, Akshay Krishnamurthy, and Aarti Singh. Noise thresholds
for spectral clustering. Advances in Neural Information Processing Systems, 24:954–962, 2011.
[6] V´eronique Barriel and Pascal Tassy. Rooting with multiple outgroups: consensus versus par-
simony. Cladistics, 14(2):193–200, 1998.
[7] Jon Louis Bentley, Dorothea Haken, and James B Saxe. A general method for solving divide-
and-conquer recurrences. ACM SIGACT News, 12(3):36–44, 1980.
[8] Laura M Boykin, Laura Salter Kubatko, and Timothy K Lowrey. Comparison of methods for
rooting phylogenetic trees: A case study using orcuttieae (poaceae: Chloridoideae). Molecular
Phylogenetics and Evolution, 54(3):687–700, 2010.
39

## Page 40

[9] Joseph T Chang. Full reconstruction of markov models on evolutionary trees: identiﬁability
and consistency. Mathematical Biosciences, 137(1):51–73, 1996.
[10] Myung Jin Choi, Vincent YF Tan, Animashree Anandkumar, and Alan S Willsky. Learning
latent tree graphical models. Journal of Machine Learning Research, 12:1771–1812, 2011.
[11] Douglas E Crabtree. Applications of m-matrices to non-negative matrices. Duke Mathematical
Journal, 33(1):197–208, 1966.
[12] Chris HQ Ding, Xiaofeng He, Hongyuan Zha, Ming Gu, and Horst D Simon. A min-max cut
algorithm for graph partitioning and data clustering. In Proceedings 2001 IEEE International
Conference on Data Mining, pages 107–114. IEEE, 2001.
[13] Florian Dorﬂer and Francesco Bullo. Kron reduction of graphs with applications to electrical
networks. IEEE Transactions on Circuits and Systems I: Regular Papers, 60(1):150–163, 2012.
[14] Richard Durbin, Sean R Eddy, Anders Krogh, and Graeme Mitchison. Biological sequence
analysis: probabilistic models of proteins and nucleic acids. Cambridge University Press, 1998.
[15] P´eter L Erd˝os, Michael A Steel, L´aszl´o A Sz´ekely, and Tandy J Warnow. A few logs suﬃce to
build (almost) all trees (i). Random Structures & Algorithms, 14(2):153–184, 1999.
[16] Nicholas Eriksson.
Tree construction using singular value decomposition.
New York, NY:
Cambridge University Press, 2005., pages 347–358, 2005.
[17] George F Estabrook, FR McMorris, and Christopher A Meacham. Comparison of undirected
phylogenetic trees based on subtrees of four evolutionary units. Systematic Zoology, 34(2):193–
200, 1985.
[18] Joseph Felsenstein. Inferring phylogenies, volume 2. Sinauer Associates, 2003.
[19] Miroslav Fiedler. A property of eigenvectors of nonnegative symmetric matrices and its appli-
cation to graph theory. Czechoslovak Mathematical Journal, 25(4):619–633, 1975.
[20] Alexander Griﬃng. Connections between numerical taxonomy and phylogenetics. Ph.D Thesis,
2012.
[21] St´ephane Guindon and Olivier Gascuel. A simple, fast, and accurate algorithm to estimate
large phylogenies by maximum likelihood. Systematic Biology, 52(5):696–704, 2003.
[22] Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp. Finding structure with randomness:
Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review,
53(2):217–288, 2011.
[23] Stefan Harmeling and Christopher KI Williams. Greedy learning of binary latent trees. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 33(6):1087–1097, 2010.
[24] Masami Hasegawa, Hirohisa Kishino, and Taka-aki Yano. Dating of the human-ape splitting
by a molecular clock of mitochondrial DNA. Journal of Molecular Evolution, 22(2):160–174,
1985.
40

## Page 41

[25] David M Hillis, Craig Moritz, Barbara K Mable, and Richard G Olmstead. Molecular system-
atics, volume 23. Sinauer Associates Sunderland, MA, 1996.
[26] Ariel Jaﬀe, Noah Amsel, Yariv Aizenbud, Boaz Nadler, Joseph T Chang, and Yuval Kluger.
Spectral neighbor joining for reconstruction of latent tree models. SIAM Journal on Mathe-
matics of Data Science, 3(1):113–141, 2021.
[27] Ariel Jaﬀe, Yuval Kluger, Oﬁr Lindenbaum, Jonathan Patsenker, Erez Peterfreund, and Stefan
Steinerberger. The spectral underpinning of word2vec. Frontiers in Applied Mathematics and
Statistics, 6:64, 2020.
[28] Tao Jiang, Paul Kearney, and Ming Li. A polynomial time approximation scheme for inferring
evolutionary trees from quartet topologies and its application. SIAM Journal on Computing,
30(6):1942–1961, 2001.
[29] Matthew G Jones, Alex Khodaverdian, Jeﬀrey J Quinn, Michelle M Chan, Jeﬀrey A Hussmann,
Robert Wang, Chenling Xu, Jonathan S Weissman, and Nir Yosef. Inference of single-cell
phylogenies from lineage tracing data using cassiopeia. Genome Biology, 21:1–27, 2020.
[30] Nick S Jones and John Moriarty. Evolutionary inference for function-valued traits: Gaussian
process regression on phylogenies. Journal of The Royal Society Interface, 10(78):20120616,
2013.
[31] Irene Keller, Douda Bensasson, and Richard A Nichols. Transition-transversion bias is not
universal: a counter example from grasshopper pseudogenes. PLoS Genet, 3(2):e22, 2007.
[32] Tonny Kinene, J Wainaina, Solomon Maina, and LM Boykin. Rooting trees, methods for.
Encyclopedia of Evolutionary Biology, pages 489—-493, 2016.
[33] Mark Kot. Stochastic birth and death processes. New York, NY: Cambridge University Press,
pages 25–42, 2001.
[34] Sudhir Kumar. Molecular clocks: four decades of evolution. Nature Reviews Genetics, 6(8):654–
662, 2005.
[35] Motomu Matsui and Wataru Iwasaki. Graph splitting: a graph-based approach for superfamily-
scale phylogenetic tree reconstruction. Systematic Biology, 69(2):265–279, 2020.
[36] Radu Mihaescu, Dan Levy, and Lior Pachter.
Why neighbor-joining works. Algorithmica,
54(1):1–24, 2009.
[37] Siavash Mirarab and Tandy Warnow. Astral-ii: coalescent-based species tree estimation with
many hundreds of taxa and thousands of genes. Bioinformatics, 31(12):i44–i52, 2015.
[38] Erin K Molloy and Tandy Warnow. Statistically consistent divide-and-conquer pipelines for
phylogeny estimation using NJMerge. Algorithms for Molecular Biology, 14(1):14, 2019.
[39] Erin K Molloy and Tandy Warnow. TreeMerge: A new method for improving the scalability
of species tree estimation methods. Bioinformatics, 35(14):i417–i426, 2019.
41

## Page 42

[40] Benoit Morel, Pierre Barbera, Lucas Czech, Ben Bettisworth, Lukas H¨ubner, Sarah Lutteropp,
Dora Serdari, Evangelia-Georgia Kostaki, Ioannis Mamais, Alexey M Kozlov, et al. Phyloge-
netic analysis of sars-cov-2 data is diﬃcult. Molecular biology and evolution, 38(5):1777–1791,
2021.
[41] Elchanan Mossel and S´ebastien Roch. Learning nonsingular phylogenies and hidden markov
models. In Proceedings of the thirty-seventh annual ACM symposium on Theory of computing,
pages 366–375, 2005.
[42] Rapha¨el Mourad, Christine Sinoquet, Nevin Lianwen Zhang, Tengfei Liu, and Philippe Leray.
A survey on latent tree models and applications. Journal of Artiﬁcial Intelligence Research,
47:157–203, 2013.
[43] Masatoshi Nei and Sudhir Kumar. Molecular evolution and phylogenetics. Oxford University
Press, 2000.
[44] Morgan N Price, Paramvir S Dehal, and Adam P Arkin. FastTree 2-approximately maximum-
likelihood trees for large alignments. PLoS ONE, 5(3):e9490, 2010.
[45] Jeﬀrey J Quinn, Matthew G Jones, Ross A Okimoto, Shigeki Nanjo, Michelle M Chan, Nir
Yosef, Trever G Bivona, and Jonathan S Weissman. Single-cell lineages reveal the rates, routes,
and drivers of metastasis in cancer xenografts. Science, 2021.
[46] Naruya Saitou and Masatoshi Nei. The neighbor-joining method: a new method for recon-
structing phylogenetic trees. Molecular Biology and Evolution, 4(4):406–425, 1987.
[47] Michael J Sanderson and Amy C Driskell. The challenge of constructing large phylogenetic
trees. Trends in plant science, 8(8):374–379, 2003.
[48] Charles Semple, Mike Steel, et al.
Phylogenetics, volume 24.
Oxford University Press on
Demand, 2003.
[49] Julia Sigwart. Coalescent Theory: An Introduction. Systematic Biology, 58(1):162–165, 03
2009.
[50] Kamen P Simeonov, China N Byrns, Megan L Clark, Robert J Norgard, Beth Martin, Ben Z
Stanger, Jay Shendure, Aaron McKenna, and Christopher J Lengner. Single-cell lineage tracing
of metastatic cancer reveals selection of hybrid emt states. Cancer Cell, 2021.
[51] Robert R Sokal. A statistical method for evaluating systematic relationships. Univ. Kansas,
Sci. Bull., 38:1409–1438, 1958.
[52] Alexandros Stamatakis. RAxML version 8: a tool for phylogenetic analysis and post-analysis
of large phylogenies. Bioinformatics, 30(9):1312–1313, 2014.
[53] Gilbert W Stewart. Matrix Algorithms: Volume II: Eigensystems. SIAM, 2001.
[54] Eric A Stone and Alexander R Griﬃng. On the Fiedler vectors of graphs that arise from trees
by schur complementation of the Laplacian. Linear Algebra and its Applications, 431(10):1869–
1880, 2009.
42

## Page 43

[55] Korbinian Strimmer and Arndt Von Haeseler. Quartet puzzling: a quartet maximum-likelihood
method for reconstructing tree topologies. Molecular Biology and Evolution, 13(7):964–969,
1996.
[56] Koichiro Tamura, Masatoshi Nei, and Sudhir Kumar. Prospects for inferring very large phylo-
genies by using the neighbor-joining method. Proceedings of the National Academy of Sciences,
101(30):11030–11035, 2004.
[57] Ulrike Von Luxburg. A tutorial on spectral clustering. Statistics and Computing, 17(4):395–
416, 2007.
[58] Tandy Warnow.
Supertree construction:
opportunities and challenges.
arXiv preprint
arXiv:1805.03530, 2018.
[59] Mark Wilkinson, James O McInerney, Robert P Hirt, Peter G Foster, and T Martin Embley.
Of clades and clans: terms for phylogenetic relationships in unrooted trees. Trends in Ecology
and Evolution, 22(10.1016), 2007.
[60] Ziheng Yang and Bruce Rannala. Molecular phylogenetics: principles and practice. Nature
reviews genetics, 13(5):303, 2012.
[61] Yi Yu, Tengyao Wang, and Richard J Samworth. A useful variant of the Davis–Kahan theorem
for statisticians. Biometrika, 102(2):315–323, 2015.
[62] Nevin L Zhang, Shihong Yuan, Tao Chen, and Yi Wang. Latent tree models and diagnosis in
traditional chinese medicine. Artiﬁcial Intelligence in Medicine, 42(3):229–245, 2008.
[63] Shu-Bo Zhang, Song-Yu Zhou, Jian-Guo He, and Jian-Huang Lai. Phylogeny inference based
on spectral graph clustering. Journal of Computational Biology, 18(4):627–637, 2011.
[64] Xiaofan Zhou, Xing-Xing Shen, Chris Todd Hittinger, and Antonis Rokas. Evaluating fast
maximum likelihood-based phylogenetic programs using empirical phylogenomic data sets.
Molecular Biology and Evolution, 35(2):486–503, 2018.
43

## Page 44

x1
x2
x3
x4
h0
h1
1/4
1/8
1/8
1/8
1/8
1/4
(a) A graph G0 that is a multiplicative sub-graph
with respect to T and the the terminal nodes of T .
h0
h1
x1
x2
x3
x4
1/2
1/2
1/2
1/2
1/2
(b) A tree T0 with 4 terminal nodes and 2 internal
nodes. The weight over all edges is equal to 1/2.
x1
x2
x3
x4
h0
h1
3/4
3/4
3/8
3/8
3/16
(c) A graph G1, created by adding the node h0 to G0.
The weights of the graph are set such that LG0 =
LG1/h0.
h0
h1
x1
x2
x3
x4
3/4
3/4
√
3/4
√
3/4
√
3/2
(d) A tree T1 with weights set such that G1 is
a multiplicative subgraph of T1
with respect to
{x1, x2, x3, x4, h1}
h0
h1
x1
x2
x3
x4
3/4
3/4
3/4
3/4
3/2
(e) A tree T2 with weights set such that LT2/h2 =
LG1.
Figure 11: Constructing a tree T2 such that the Schur complement of its Laplacian with respect to the
internal nodes is equal to LG.
44

## Page 45

400
600
800
1000 1200
Number of samples (n)
0.0
0.2
0.4
0.6
0.8
1.0
normalized RF distance
400
600
800
1000 1200
Number of samples (n)
101
102
103
runtime (s)
STDR+RAxML (64)
RAxML
STDR+NJ (64)
NJ
STDR+SNJ (64)
SNJ
Figure 12: A caterpillar tree with m = 512 terminal nodes. The mean and standard deviation of the runtime
(right) and RF distance between the reconstructed tree and the input tree (left) are shown for each method
over 5 independent runs.
2000
3000
4000
5000
6000
7000
8000
Number of samples (n)
0.15
0.20
0.25
0.30
0.35
normalized RF distance
2000
3000
4000
5000
6000
7000
8000
Number of samples (n)
500
1000
1500
2000
runtime (s)
STDR+RAxML (128)
treemerge
Figure 13: A coalesent tree with m = 2000 terminal nodes.
The mean and standard deviation of the
normalized RF distance (left) between the reconstructed tree and the input tree and of the runtime (right)
are shown for each method over 5 independent runs.
45
