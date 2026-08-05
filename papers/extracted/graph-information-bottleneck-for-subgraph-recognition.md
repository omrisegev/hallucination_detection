---
source_pdf: papers/Graph Information Bottleneck for Subgraph Recognition.pdf
slug: graph-information-bottleneck-for-subgraph-recognition
pages: 13
extracted_on: 2026-08-05
---

# Graph Information Bottleneck for Subgraph Recognition

## Page 1

Graph Information Bottleneck for Subgraph Recognition
Junchi Yu1,2,3, Tingyang Xu3, Yu Rong3, Yatao Bian3, Junzhou Huang3, and Ran He1,2,4
1NLPR&CRIPAC, Institute of Automation, Chinese Academy of Sciences
2University of Chinese Academy of Sciences
3Tencent AI LAB
4Center for Excellence in Brain Science and Intelligence Technology, CAS
October 13, 2020
Abstract
Given the input graph and its label/property, several key problems of graph learning, such as
ﬁnding interpretable subgraphs, graph denoising and graph compression, can be attributed to the
fundamental problem of recognizing a subgraph of the original one. This subgraph shall be as infor-
mative as possible, yet contains less redundant and noisy structure. This problem setting is closely
related to the well-known information bottleneck (IB) principle, which, however, has less been studied
for the irregular graph data and graph neural networks (GNNs). In this paper, we propose a frame-
work of Graph Information Bottleneck (GIB) for the subgraph recognition problem in deep graph
learning. Under this framework, one can recognize the maximally informative yet compressive sub-
graph, named IB-subgraph. However, the GIB objective is notoriously hard to optimize, mostly due
to the intractability of the mutual information of irregular graph data and the unstable optimization
process. In order to tackle these challenges, we propose: i) a GIB objective based-on a mutual infor-
mation estimator for the irregular graph data; ii) a bi-level optimization scheme to maximize the GIB
objective; iii) a connectivity loss to stabilize the optimization process. We evaluate the properties
of the IB-subgraph in three application scenarios: improvement of graph classiﬁcation, graph inter-
pretation and graph denoising. Extensive experiments demonstrate that the information-theoretic
IB-subgraph enjoys superior graph properties.
1
Introduction
Classifying the underlying labels or properties of graphs is a fundamental problem in deep graph learning
with applications across many ﬁelds, such as biochemistry and social network analysis. However, real
world graphs are likely to contain redundant even noisy information [9, 36], which poses a huge negative
impact for graph classiﬁcation. This triggers an interesting problem of recognizing an informative yet
compressed subgraph from the original graph. For example, in drug discovery, when viewing molecules
as graphs with atoms as nodes and chemical bonds as edges, biochemists are interested in identifying
the subgraphs that mostly represent certain properties of the molecules, namely the functional groups
[17, 11]. In graph representation learning, the predictive subgraph highlights the vital substructure for
graph classiﬁcation, and provides an alternative way for yielding graph representation besides mean/sum
aggregation [19, 30, 32] and pooling aggregation [35, 21, 4]. In graph attack and defense, it is vital to
purify a perturbed graph and mine the robust structures for classiﬁcation [16].
Recently, the mechanism of self-attentive aggregation [22] somehow discovers a vital substructure at
node level with a well-selected threshold. However, this method only identiﬁes isolated important nodes
but ignores the topological information at subgraph level. Consequently, it leads to a novel challenge as
subgraph recognition: How can we recognize a compressed subgraph with minimum information loss in
terms of predicting the graph labels/properties?
Recalling the above challenge, there is a similar problem setting in information theory called infor-
mation bottleneck (IB) principle [29], which aims to juice out a compressed data from the original data
that keeps most predictive information of labels or properties. Enhanced with deep learning, IB can learn
informative representation from regular data in the ﬁelds of computer vision [24, 1, 23], reinforcement
learning [13, 15] and natural language precessing [31]. However, current IB methods, like VIB [1], is still
incapable for irregular graph data. It is still challenging for IB to compress irregular graph data, like a
subgraph from an original graph, with a minimum information loss.
Hence, we advance the IB principle for irregular graph data to resolve the proposed subgraph recogni-
tion problem, which leads to a novel principle, Graph Information Bottleneck (GIB). Diﬀerent from prior
researches in IB that aims to learn an optimal representation of the input data in the hidden space, GIB
directly reveals the vital substructure in the subgraph level. We ﬁrst i) leverage the mutual information
1
arXiv:2010.05563v1  [cs.LG]  12 Oct 2020

## Page 2

estimator from Deep Variational Information Bottleneck (VIB) [1] for irregular graph data as the GIB
objective. However, VIB is intractable to compute the mutual information without knowing the distri-
bution forms, especially on graph data. To tackle this issue, ii) we adopt a bi-level optimization scheme
to maximize the GIB objective. Meanwhile, the continuous relaxation that we adopt to approach the
discrete selection of subgraph will lead to unstable optimization process. To further stabilize the training
process and encourage a compact subgraph, iii) we propose a novel connectivity loss to assist GIB to
eﬀectively discover the maximally informative but compressed subgraph, which is deﬁned as IB-subgraph.
By optimizing the above GIB objective and connectivity loss, one can recognize the IB-subgraph with-
out any explicit subgraph annotation. On the other hand, iv) GIB is model-agnostic and can be easily
plugged into various Graph Neural Networks (GNNs).
We evaluate the properties of the IB-subgraph in three application scenarios: improvement of graph
classiﬁcation, graph interpretation, and graph denoising. Extensive experiments on both synthetic and
real world datasets demonstrate that the information-theoretic IB-subgraph enjoys superior graph prop-
erties compared to the subgraphs found by SOTA baselines.
2
Related Work
Graph Classiﬁcation. In recent literature, there is a surge of interest in adopting graph neural net-
works (GNN) in graph classiﬁcation. The core idea is to aggregate all the node information for graph
representation. A typical implementation is the mean/sum aggregation [19, 32], which is to average or
sum up the node embeddings. An alternative way is to leverage the hierarchical structure of graphs,
which leads to the pooling aggregation [35, 38, 21, 4]. When tackling with the redundant and noisy
graphs, these approaches will likely to result in sub-optimal graph representation.
Information Bottleneck. Information bottleneck (IB), originally proposed for signal processing,
attempts to ﬁnd a short code of the input signal but preserve maximum information of the code [29]. [1]
ﬁrstly bridges the gap between IB and the deep learning, and proposed variational information bottleneck
(VIB). Nowadays, IB and VIB have been wildly employed in computer vision [24, 23], reinforcement
learning [13, 15], natural language processing [31] and speech and acoustics [25] due to the capability of
learning compact and meaningful representations. However, IB is less researched on irregular graphs due
to the intractability of mutual information.
Subgraph Discovery. Traditional subgraph discovery includes dense subgraph discovery and fre-
quent subgraph mining. Dense subgraph discovery aims to ﬁnd the subgraph with the highest density
(e.g. the number of edges over the number of nodes [8, 12]). Frequent subgraph mining is to look for
the most common substructure among graphs [33, 18, 37]. Recently, researchers discover the vital sub-
structure at node level via the attention mechanism [30, 21, 20]. [34] further identiﬁes the important
computational graph for node classiﬁcation. [2] discovers subgraph representations with speciﬁc topology
given subgraph-level annotation.
3
Notations and Preliminaries
Let {(G1, Y1), . . . , (GN, YN)} be a set of N graphs with their real value properties or categories, where
Gn refers to the n-th graph and Yn refers to the corresponding properties or labels. We denote by Gn =
(V, E, A, X) the n-th graph of size Mn with node set V = {Vi|i = 1, . . . , Mn}, edge set E = {(Vi, Vj)|i >
j; Vi, Vj is connected}, adjacent matrix A ∈{0, 1}Mn×Mn, and feature matrix X ∈RMn×d of V with d
dimensions, respectively. Denote the neighborhood of Vi as N(Vi) = {Vj|(Vi, Vj) ∈E}. We use Gsub as a
speciﬁc subgraph and Gsub as the complementary structure of Gsub in G. Let f : G →R/[0, 1, · · · , n] be
the mapping from graphs to the real value property or category, Y , G is the domain of the input graphs.
I(X, Y ) refers to the Shannon mutual information of two random variables.
3.1
Graph convolutional network
Graph convolutional network (GCN) is widely adopted to graph classiﬁcation. Given a graph G = (V, E)
with node feature X and adjacent matrix A, GCN outputs the node embeddings X
′ from the following
process:
X
′ = GCN(A, X; W ) = ReLU(D−1
2 ˆ
AD−1
2 W ),
(1)
where D refers to the diagonal matrix with nodes’ degrees and W refers to the model parameters.
One can simply sum up the node embeddings to get a ﬁxed length graph embeddings [32]. Recently,
researchers attempt to exploit hierarchical structure of graphs, which leads to various graph pooling
methods [22, 10, 21, 6, 38, 26, 35]. [22] enhances the graph pooling with self-attention mechanism to
leverage the importance of diﬀerent nodes contributing to the results. Finally, the graph embedding is
2

## Page 3

Figure 1: Illustration of the proposed graph information bottleneck (GIB) framework. We employ a
bi-level optimization scheme to optimize the GIB objective and thus yielding the IB-subgraph. In the
inner optimization phase, we estimate I(G, Gsub) by optimizing the statistics network of the DONSKER-
VARADHAN representation [7]. Given a good estimation of I(G, Gsub), in the outer optimization phase,
we maximize the GIB objective by optimizing the mutual information, the classiﬁcation loss Lcls and
connectivity loss Lcon.
obtained by multiplying the node embeddings with the normalized attention scores:
E = Att(X
′) = softmax(Φ2tanh(Φ1X
′T ))X
′,
(2)
where Φ1 and Φ2 refers to the model parameters of self-attention.
3.2
Optimizing Information bottleneck objective
Given the input signal X and the label Y , the objective of IB is maximized to ﬁnd the the internal
code Z: maxZ I(Z, Y ) −βI(X, Z), where β refers to a hyper-parameter trading oﬀinformativeness and
compression. Optimizing this objective will lead to a compact but informative Z. [1] optimize a tractable
lower bound of the IB objective:
LV IB = 1
N
XN
i=1
Z
p(z|xi) log qφ(yi|z)dz −βKL(p(z|xi)|r(z)),
(3)
where qφ(yi|z) is the variational approximation to pφ(yi|z) and r(z) is the prior distribution of Z. However,
it is hard to estimate the mutual information in high dimensional space when the distribution forms are
inaccessible, especially for irregular graph data.
4
Optimizing the Graph Information Bottleneck Objective for
Subgraph Recognition
In this section, we will elaborate the proposed method in details. We ﬁrst formally deﬁne the graph
information bottleneck and IB-subgraph. Then, we introduce a novel framework for GIB to eﬀectively
ﬁnd the IB-subgraph. We further propose a bi-level optimization scheme and a graph mutual information
estimator for GIB optimization. Moreover, we do a continuous relaxation to the generation of subgraph,
and propose a novel loss to stabilize the training process.
4.1
graph information bottleneck
We generalize the information bottleneck principle to learn a informative representation of irregular
graphs, which leads to the graph information bottleneck (GIB) principle.
Deﬁnition 4.1 (Graph Information Bottleneck). Given a graph G and its label Y , the GIB seeks for the
most informative yet compressed representation Z by optimizing the following objective:
max
Z
I(Y, Z) s.t. I(G, Z) ≤Ic.
(4)
where Ic is the information constraint between G and Z. By introducing a Lagrange multiplier β to Eq. 4,
we reach its unconstrained form:
max
Z
I(Y, Z) −βI(G, Z).
(5)
3

## Page 4

Eq. 5 gives a general formulation of GIB. Here, in subgraph recognition, we focus on a subgraph which
is compressed with minimum information loss in terms of graph properties.
Deﬁnition 4.2 (IB-subgraph). For a graph G, its maximally informative yet compressed subgraph,
namely IB-subgraph can be obtained by optimizing the following objective:
max
Gsub∈Gsub I(Y, Gsub) −βI(G, Gsub).
(6)
where Gsub indicates the set of all subgraphs of G.
IB-subgraph enjoys various pleasant properties and can be applied to multiple graph learning tasks
such as improvement of graph classiﬁcation, graph interpretation, and graph denoising. However, the
GIB objective in Eq. 6 is notoriously hard to optimize due to the intractability of mutual information
and the discrete nature of irregular graph data. We then introduce approaches on how to optimize such
objective and derive the IB-subgraph.
4.2
Bi-level optimization for the GIB objective
The GIB objective in Eq. 6 consists of two parts. We examine the ﬁrst term I(Y, Gsub) in Eq. 6, ﬁrst.
This term measures the relevance between Gsub and Y . We expand I(Y, Gsub) as:
I(Y, Gsub) =
Z
p(y, Gsub) log p(y|Gsub)dy dGsub + H(Y ).
(7)
H(Y ) is the entropy of Y and thus can be ignored.
In practice, we approximate p(y, Gsub) with an
empirical distribution p(y, Gsub) ≈
1
N
PN
i=1 δyi(y)δGsub,i(Gsub), where Gsub is the output subgraph and
Y is the graph label.
By substituting the true posterior p(y|Gsub) with a variational approximation
qφ1(y|Gsub), we obtain a tractable lower bound of the ﬁrst term in Eq. 6:
I(Y, Gsub) ≥
Z
p(y, Gsub) log qφ1(y|Gsub)dy dGsub
≈1
N
N
X
i=1
qφ1(yi|Gsubi) =: −Lcls(qφ1(y|Gsub), ygt),
(8)
where ygt is the ground truth label of the graph. Eq. 8 indicates that maximizing I(Y, Gsub) is achieved
by the minimization of the classiﬁcation loss between Y and Gsub as Lcls. Intuitively, minimizing Lcls
encourages the subgraph to be predictive of the graph label. In practice, we choose the cross entropy loss
for categorical Y and the mean squared loss for continuous Y , respectively. For more details of deriving
Eq. 7 and Eq. 8, please refer to Appendix A.1.
Then, we consider the minimization of I(G, Gsub) which is the second term of Eq. 6. Remind that [1]
introduces a tractable prior distribution r(Z) in Eq. 3, and thus results in a variational upper bound.
However, this setting is troublesome as it is hard to ﬁnd a reasonable prior distribution for p(Gsub), which
is the distribution of graph substructures instead of latent representation. Thus we go for another route.
Directly applying the DONSKER-VARADHAN representation [7] of the KL-divergence, we have:
I(G, Gsub) =
sup
fφ2:G×G→R
EG,Gsub∈p(G,Gsub)fφ2(G, Gsub) −log EG∈p(G),Gsub∈p(Gsub)efφ2(G,Gsub),
(9)
where fφ2 is the statistics network that maps from the graph set to the set of real numbers. In order to
approximate I(G, Gsub) using Eq. 9, we design a statistics network based on modern GNN architectures
as shown by Figure 1: ﬁrst we use a GNN to extract embeddings from both G and Gsub (parameter
shared with the subgraph generator, which will be elaborated in Section 4.3), then concatenate G and
Gsub embeddings and feed them into a MLP, which ﬁnally produces the real number. In conjunction with
the sampling method to approximate p(G, Gsub), p(G) and p(Gsub), we reach the following optimization
problem to approximate1 I(G, Gsub):
max
φ2
LMI(φ2, Gsub) = 1
N
N
X
i=1
fφ2(Gi, Gsub,i) −log 1
N
N
X
i=1,j̸=i
efφ2(Gi,Gsub,j).
(10)
With the approximation to the MI in graph data, we combine Eq. 6 , Eq. 8 and Eq. 10 and formulate
the optimization process of GIB as a tractable bi-level optimization problem:
min
Gsub,φ1
L(Gsub, φ1, φ∗
2) = Lcls(qφ1(y|Gsub), ygt) + βLMI(φ∗
2, Gsub)
(11)
s.t.
φ∗
2 = arg max
φ2
LMI(φ2, Gsub).
(12)
1Notice that the MINE estimator [3] straightforwardly uses the DONSKER-VARADHAN representation to derive an
MI estimator between the regular input data and its vectorized representation/encoding. It cannot be applied to estimate
the mutual information between G and Gsub since both of G and Gsub are irregular graph data.
4

## Page 5

We ﬁrst derive a sub-optimal φ2 notated as φ∗
2 by optimizing Eq. 12 for T steps as inner loops. After the
T-step optimization of the inner-loop ends, Eq. 10 is a proxy for MI minimization for the GIB objective
as an outer loop. Then, the parameter φ1 and the subgraph Gsub are optimized to yield IB-subgraph.
However, in the outer loop, the discrete nature of G and Gsub hinders applying the gradient-based method
to optimize the bi-level objective and ﬁnd the IB-subgraph.
4.3
The Subgraph Generator and connectivity loss
To alleviate the discreteness in Eq. 11, we propose the continuous relaxation to the subgraph recognition
and propose a loss to stabilize the training process.
Subgraph generator: For the input graph G, we generate its IB-subgraph with the node assignment
S which indicates the node is in Gsub or Gsub. Then, we introduce a continuous relaxation to the node
assignment with the probability of nodes belonging to the Gsub or Gsub. For example, the i-th row of S
is 2-dimensional vector [p(Vi ∈Gsub|Vi), p(Vi ∈Gsub|Vi)]. We ﬁrst use an l-layer GNN to obtain the node
embedding and employ a multi-layer perceptron (MLP) to output S :
Xl = GNN(A, Xl−1; θ1),
S
= MLP(Xl; θ2).
(13)
S is a n × 2 matrix, where n is the number of nodes. For simplicity, we compile the above modules as
the subgraph generator, denoted as g(; θ) with θ := (θ1, θ2). When S is well-learned, the assignment of
nodes is supposed to saturate to 0/1. The representation of Gsub, which is employed for predicting the
graph label, can be obtained by taking the ﬁrst row of ST Xl.
Connectivity loss: However, poor initialization will cause p(Vi ∈Gsub|Vi) and p(Vi ∈Gsub|Vi) to be
close. This will either lead the model to assign all nodes to Gsub / Gsub, or result that the representations
of Gsub contain much information from the redundant nodes. These two scenarios will cause the training
process to be unstable. On the other hand, we suppose our model to have an inductive bias to better
leverage the topological information while S outputs the subgraph at a node-level. Therefore, we propose
the following connectivity loss:
Lcon = ||Norm(ST AS) −I2||F ,
(14)
where Norm(·) is the row-wise normalization, || · ||F is the Frobenius norm, and I2 is a 2 × 2 identity
matrix. Lcon not only leads to distinguishable node assignment, but also encourage the subgraph to be
compact. Take (ST AS)1: for example, denote a11, a12 the element 1,1 and the element 1,2 of ST AS,
a11 =
X
i,j
Aijp(Vi ∈Gsub|Vi)p(Vj ∈Gsub|Vj),
a12 =
X
i,j
Aijp(Vi ∈Gsub|Vi)p(Vj ∈Gsub|Vj).
(15)
Minimizing Lcon results in
a11
a11+a12 →1. This occurs if Vi is in Gsub, the elements of N(Vi) have a high
probability in Gsub. Minimizing Lcon also encourages
a12
a11+a12 →0. This encourages p(Vi ∈Gsub|Vi) →0/1
and less cuts between Gsub and Gsub. This also holds for Gsub when analyzing a21 and a22.
In a word, Lcon encourages distinctive S to stabilize the training process and a compact topology in
the subgraph. Therefore, the overall loss is:
min
θ,φ1
L(θ, φ1, φ∗
2) = Lcon(g(G; θ)) + Lcls(qφ1(g(G; θ)), ygt) + βLMI(φ∗
2, Gsub)
s.t.
φ∗
2 = arg max
φ2
LMI(φ2, Gsub).
(16)
We provide the pseudo code in the Appendix to better illustrate how to optimize the above objective.
5
Experiments
In this section, we evaluate the proposed GIB method on three scenarios, including improvement of graph
classiﬁcation, graph interpretation and graph denoising.
5.1
Baselines and settings
Improvement of graph classiﬁcation: For improvement of graph classiﬁcation, GIB generates graph
representation by aggregating the subgraph information. We plug GIB into various backbones including
GCN [19], GAT [30], GIN [32] and GraphSAGE [14].
We compare the proposed method with the
mean/sum aggregation [19, 30, 14, 32] and pooling aggregation [38, 26, 35, 6] in terms of classiﬁcation
accuracy.
5

## Page 6

Table 1: Classiﬁcation accuracy in percent. The pooling methods yield pooling aggregation while the
backbones yield mean aggregation. The proposed GIB method with backbones yields subgraph embedding
by aggregating the nodes in subgraphs.
Method
MUTAG
PROTEINS
IMDB-BINARY
DD
SortPool
0.844 ± 0.141
0.747 ± 0.044
0.712 ± 0.047
0.732 ± 0.087
ASAPool
0.743 ± 0.077
0.721 ± 0.043
0.715 ± 0.044
0.717 ± 0.037
DiﬀPool
0.839 ± 0.097
0.727 ± 0.046
0.709 ± 0.053
0.778 ± 0.030
EdgePool
0.759 ± 0.077
0.723 ± 0.044
0.728 ± 0.044
0.736 ± 0.040
AttPool
0.721 ± 0.086
0.728 ± 0.041
0.722 ± 0.047
0.711 ± 0.055
GCN
0.743±0.110
0.719±0.041
0.707 ± 0.037
0.725 ± 0.046
GraphSAGE
0.743±0.077
0.721 ± 0.042
0.709 ± 0.041
0.729 ± 0.041
GIN
0.825±0.068
0.707 ± 0.056
0.732 ± 0.048
0.730 ± 0.033
GAT
0.738 ± 0.074
0.714 ± 0.040
0.713 ± 0.042
0.695 ± 0.045
GCN+GIB
0.776 ± 0.075
0.748 ± 0.046
0.722 ± 0.039
0.765 ± 0.050
GraphSAGE+GIB
0.760 ± 0.074
0.734 ± 0.043
0.719 ± 0.052
0.781 ± 0.042
GIN+GIB
0.839 ± 0.064
0.749 ± 0.051
0.737 ± 0.070
0.747 ± 0.039
GAT+GIB
0.749 ± 0.097
0.737 ± 0.044
0.729 ± 0.046
0.769 ± 0.040
Table 2: The mean and variance of absolute property bias between the graphs and the corresponding
subgraphs. Note that we try several initiations for GCN+GIB w/o Lcon and LMI to get the current
results due to the instability of optimization process.
Method
QED
DRD2
HLM-CLint
MLM-CLint
GCN+Att05
0.48± 0.07
0.20± 0.13
0.90± 0.89
0.92± 0.61
GCN+Att07
0.41± 0.07
0.16± 0.11
1.18± 0.60
1.69± 0.88
GCN+GIB w/o Lcon
0.46± 0.07
0.15± 0.12
0.45± 0.37
1.58± 0.86
GCN+GIB w/o LMI
0.43± 0.15
0.21± 0.13
0.48± 0.34
1.20± 0.97
GCN+GIB
0.38± 0.12
0.06± 0.09
0.37± 0.30
0.72± 0.55
Graph interpretation: The goal of graph interpretation is to ﬁnd the substructure which shares
the most similar property to the molecule. If the substructure is disconnected, we evaluate its largest
connected part. We compare GIB with the attention mechanism [22]. That is, we attentively aggregate
the node information for graph prediction. The interpretable subgraph is generated by choosing the nodes
with top 50% and 70% attention scores, namely Att05 and Att07. GIB outputs the interpretation with the
IB-subgraph. Then, we evaluate the absolute property bias (the absolute value of the diﬀerence between
the property of graph and subgraph) between the graph and its interpretation. For fare comparisons, we
adopt the same GCN as the backbone for diﬀerent methods.
Graph denoising: We translate the permuted graph into the line-graph and use GIB and attention
to 1) infer the real structure of graph, 2) classify the permuted graph via the inferred structure. We
further compare the performance of GCN and DiﬀPool on the permuted graphs.
5.2
datasets
Improvement of graph classiﬁcation: We evaluate diﬀerent methods on the datasets of MUTAG
[28], PROETINS [5], IMDB-BINARY and DD [27] datasets. 2.
Graph interpretation: We construct the datasets for graph interpretation on four molecule prop-
erties based on ZINC dataset, which contains 250K molecules. QED measures the drug likeness of a
molecule, which is bounded within the range (0, 1.0). DRD2 measures the probability that a molecule is
active against dopamine type 2 receptor, which is bounded with (0, 1.0). HLM-CLint and MLM-CLint
are estimate values of in vitro human and mouse liver microsome metabolic stability (base 10 logrithm of
mL/min/g). We sample the molecules with QED ≥0.85, DRD2 ≥0.50, HLM-CLint ≥2, MLM-CLint
≥2 for each task. We use 85% of these molecules for training, 5% for validating and 10% for testing.
Graph denoising: We generate a synthetic dataset by adding 30% redundant edges for each graph
in MUTAG dataset. We use 70% of these graphs for training, 5% for validating and 25% for testing.
2We follow the protocol in https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/kernel
6

## Page 7

Figure 2: The molecules with their interpretable subgraphs discovered by diﬀerent methods.
These
subgraphs exhibit similar chemical properties compared to the molecules on the left.
Table 3: Quantitative results on graph denoising. We report the classiﬁcation accuracy (Acc), number of
real edges over total real edges (Recall) and number of real edges over total edges in subgraphs (Precision)
on the test set.
Method
GCN
DiﬀPool
GCN+Att05
GCN+Att07
GCN+GIB
Recall
-
-
0.226±0.047
0.324± 0.049
0.493± 0.035
Precision
-
-
0.638± 0.141
0.675± 0.104
0.692 ±0.061
Acc
0.617
0.658
0.649
0.667
0.684
5.3
Results
Improvement of Graph Classiﬁcation: In Table 1, we comprehensively evaluate the proposed method
and baselines on improvement of graph classiﬁcation. We train GIB on various backbones and aggregate
the graph representations only from the subgraphs.
We compare the performance of our framework
with the mean/sum aggregation and pooling aggregation.
This shows that GIB improves the graph
classiﬁcation by reducing the redundancies in the graph structure.
Table 4: Average number of disconnected substructures per
graph selected by diﬀerent methods.
Method
QED
DRD2
HLM
MLM
GCN+Att05
3.38
1.94
3.11
5.16
GCN+Att07
2.04
1.76
2.75
3.00
GCN+GIB
1.57
1.08
2.29
2.06
Graph interpretation:
Table 2
shows the quantitative performance of
diﬀerent methods on the graph inter-
pretation task.
GIB is able to gen-
erate precise graph interpretation (IB-
subgraph), as the substructures found
by GIB has the most similar property
to the input molecules. Then we derive
two variants of our method by deleting Lcon and LMI. GIB also outperforms the variants, and thus
indicates that every part of our model does contribute to the improvement of performance. In practice,
we observe that removing Lcon will lead to unstable training process due to the continuous relaxation
to the generation of subgraph. In Fig. 2, GIB generates more compact and reasonable interpretation to
the property of molecules conﬁrmed by chemical experts. More results are provided in the Appendix. In
Table 4, we compare the average number of disconnected substructures per graph since a compact sub-
graph should preserve more topological information. GIB generates more compact subgraphs to better
interpret the graph property.
Graph denoising: Table 3 shows the performance of diﬀerent methods on noisy graph classiﬁcation.
GIB outperforms the baselines on classiﬁcation accuracy by a large margin due to the superior property
of IB-subgraph. Moreover, GIB is able to better reveal the real structure of permuted graphs in terms of
precision and recall rate of true edges.
6
Conclusion
In this paper, we have studied a subgraph recognition problem to infer a maximally informative yet
compressed subgraph. We deﬁne such a subgraph as IB-subgraph and propose the graph information
bottleneck (GIB) framework for eﬀectively discovering an IB-subgraph. We derive the GIB objective from
a mutual information estimator for irregular graph data, which is optimized by a bi-level learning scheme.
A connectivity loss is further used to stabilize the learning process. We evaluate our GIB framework in
the improvement of graph classiﬁcation, graph interpretation and graph denoising. Experimental results
verify the superior properties of IB-subgraphs.
7

## Page 8

References
[1] Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, and Kevin Murphy. Deep variational information
bottleneck. In The International Conference on Representation Learning, 2017.
[2] Emily Alsentzer, Samuel G. Finlayson, Michelle M. Li, and Marinka Zitnik. Subgraph neural net-
works. 2020.
[3] Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeswar, Sherjil Ozair, Yoshua Bengio, R. Devon
Hjelm, and Aaron C. Courville. Mutual information neural estimation. In International Conference
on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 530–539, 2018.
[4] Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. Spectral clustering with graph neu-
ral networks for graph pooling. In Proceedings of the 37th International Conference on Machine
Learning, 2020.
[5] Karsten M. Borgwardt, Cheng Soon Ong, Stefan Schanauer, S. V. N. Vishwanathan, Alexander J.
Smola, and Hans-Peter Kriegel. Protein function prediction via graph kernels. In ISMB (Supplement
of Bioinformatics), pages 47–56, 2005.
[6] Frederik Diehl. Edge contraction pooling for graph neural networks. CoRR, abs/1905.10990, 2019.
[7] M. D. Donsker and S. R. S. Varadhan. Asymptotic evaluation of certain markov process expectations
for large time. Communications on Pure and Applied Mathematics, 36(2):183–212, 1983.
[8] Yixiang Fang, Kaiqiang Yu, Reynold Cheng, Laks V. S. Lakshmanan, and Xuemin Lin. Eﬃcient
algorithms for densest subgraph discovery. Proceedings of VLDB Endowment, 12(11):1719–1732,
2019.
[9] Luca Franceschi, Mathias Niepert, Massimiliano Pontil, and Xiao He. Learning discrete structures
for graph neural networks. In ICML, volume 97 of Proceedings of Machine Learning Research, pages
1972–1982. PMLR, 2019.
[10] Hongyang Gao and Shuiwang Ji. Graph u-nets. In ICML, volume 97 of Proceedings of Machine
Learning Research, pages 2083–2092. PMLR, 2019.
[11] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural
message passing for quantum chemistry. Proceedings of the 34th International Conference on Machine
Learning, 70:1263–1272, 2017.
[12] Aristides Gionis and Charalampos E. Tsourakakis. Dense subgraph discovery: Kdd 2015 tutorial.
In Knowledge Discovery and Data Mining, pages 2313–2314. ACM, 2015.
[13] Anirudh Goyal, Riashat Islam, Daniel Strouse, Zafarali Ahmed, Matthew Botvinick, Hugo
Larochelle, Yoshua Bengio, and Sergey Levine. Infobot: Transfer and exploration via the infor-
mation bottleneck. In The International Conference on Representation Learning, 2019.
[14] William L. Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large
graphs. In Advances in neural information processing systems, pages 1024–1034, 2017.
[15] Maximilian Igl, Kamil Ciosek, Yingzhen Li, Sebastian Tschiatschek, Cheng Zhang, Sam Devlin,
and Katja Hofmann.
Generalization in reinforcement learning with selective noise injection and
information bottleneck. In Advances in neural information processing systems, 2019.
[16] Wei Jin, Yao Ma, Xiaorui Liu, Xianfeng Tang, Suhang Wang, and Jiliang Tang. Graph structure
learning for robust graph neural networks. CoRR, abs/2005.10203, 2020.
[17] Wengong Jin, Regina Barzilay, and Tommi Jaakkola.
Multi-objective molecule generation using
interpretable substructures. In International Conference on Machine Learning, 2020.
[18] Nikhil S Ketkar, Lawrence Bruce Holder, and Diane Cook. Subdue: compression-based frequent
pattern discovery in graph data. In Knowledge Discovery and Data Mining, 2005.
[19] Thomas N. Kipf and Max Welling. Semi-supervised classiﬁcation with graph convolutional networks.
In The International Conference on Representation Learning, 2017.
[20] Boris Knyazev, Graham W. Taylor, and Mohamed R. Amer. Understanding attention and general-
ization in graph neural networks. In NeurIPS, pages 4204–4214, 2019.
[21] Junhyun Lee, Inyeop Lee, and Jaewoo Kang. Self-attention graph pooling. In Proceedings of the
36th International Conference on Machine Learning, 2019.
8

## Page 9

[22] Jia Li, Yu Rong, Hong Cheng, Helen Meng, Wenbing Huang, and Junzhou Huang. Semi-supervised
graph classiﬁcation: A hierarchical graph perspective. In The World Wide Wed Conference, 2019.
[23] Yawei Luo, Ping Liu, Tao Guan, Junqing Yu, and Yi Yang. Signiﬁcance-aware information bottleneck
for domain adaptive semantic segmentation. In ICCV, pages 6777–6786. IEEE, 2019.
[24] XueBin Peng, Angjoo Kanazawa, Sam Toyer, Pieter Abbeel, and Sergey Levine. Variational dis-
criminator bottleneck: Improving imitation learning, inverse rl, and gans by constraining information
ﬂow. In The International Conference on Representation Learning, 2019.
[25] Kaizhi Qian, Yang Zhang, Shiyu Chang, David Cox, and Mark Hasegawa-Johnson. Unsupervised
speech decomposition via triple information bottleneck. In Proceedings of the 37th International
Conference on Machine Learning, 2020.
[26] Ekagra Ranjan, Soumya Sanyal, and Partha Pratim Talukdar.
Asap: Adaptive structure aware
pooling for learning hierarchical graph representations. In AAAI, 2020.
[27] Ryan A. Rossi and Nesreen K. Ahmed. The network data repository with interactive graph analytics
and visualization. In AAAI, 2015.
[28] Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert Muller, and O. Anatole von Lilienfeld. Fast
and accurate modeling of molecular atomization energies with machine learning. Phys. Rev. Lett.,
108(5):058301, January 2012.
[29] Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. In
Proceedings of the 37-th Annual Allerton Conference on Communication, Control and Computing,
pages 368–377, 1999.
[30] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lia, and Yoshua
Bengio. Graph attention networks. In International Conference on Learning Representation, 2017.
[31] Rundong Wang, Xu He, Runsheng Yu, Wei Qiu, Bo An, and Zinovi Rabinovich.
Learning eﬃ-
cient multi-agent communication: An information bottleneck approach. In Proceedings of the 37th
International Conference on Machine Learning, 2020.
[32] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka.
How Powerful are Graph Neural
Networks? In Proceedings of the 7th International Conference on Learning Representations, ICLR
’19, pages 1–17, 2019.
[33] Xifeng Yan and Jiawei Yan. gspan: graph-based substructure pattern mining. In IEEE International
Conference on Data Mining, pages 721–724, 2002.
[34] Rex Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. Gnnexplainer: Gener-
ating explanations for graph neural networks. In Advances in neural information processing systems,
2019.
[35] Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, and Jure Leskovec.
Hierarchical graph representation learning with diﬀerentiable pooling. In Advances in neural infor-
mation processing systems, 2018.
[36] Donghan Yu, Ruohong Zhang, Zhengbao Jiang, Yuexin Wu, and Yiming Yang.
Graph-revised
convolutional network. CoRR, abs/1911.07123, 2019.
[37] Mohammed Javeed Zaki. Eﬃciently mining frequent embedded unordered trees. Fundamenta Infor-
mation, 66(1-2):33–52, 2005.
[38] Muhan Zhang, Zhicheng Cui, Marion Neumann, and Yixin Chen.
An end-to-end deep learning
architecture for graph classiﬁcation. In Thirty-Second AAAI Conference on Artiﬁcial Intelligence,
2018.
9

## Page 10

A
Appendix
A.1
More details about Eq. 7 and Eq. 8
Here we provide more details about how to yield Eq. 7 and Eq. 8.
I(Y, Gsub) =
Z
p(y, Gsub) log p(y|Gsub)dy dGsub −
Z
p(y, Gsub) log p(y)dy dGsub
=
Z
p(y, Gsub) log p(y|Gsub)dy dGsub + H(Y )
≥
Z
p(y, Gsub) log qφ1(y|Gsub)dy dGsub + KL(p(y|Gsub)|qφ1(y|Gsub))
≥
Z
p(y, Gsub) log qφ1(y|Gsub)dy dGsub
≈1
N
N
X
i=1
qφ1(yi|Gsubi)
= −Lcls(qφ1(y|Gsub), ygt)
(17)
A.2
case study
To understand the bi-level objective to MI minimization in Eq. 11, we provide a case study in which we
optimize the parameters of distribution to reduce the mutual information between two variables. Consider
p(x) = sign(N(0, 1)), p(y|x) = N(y; x, σ2)3. The distribution of Y is:
p(y) =
Z
p(y|x)p(x)dx
=
X
i
p(y|xi)p(xi)
= p(y|x = 1)p(x = 1) + p(y|x = −1)p(x = −1)
= 0.5(N(y; 1, σ2) + N(y; −1, σ2))
(18)
We optimize the parameter σ2 to reduce the mutual information between X and Y .
For each
epoch, we sample 20000 data points from each distribution, denoted as X = {x1, x2, · · · , x20000}, Y =
{y1, y2, · · · , y20000}. The inner-step is set to be 150. After the inner optimization ends, the model yields a
good mutual information approximator and optimize σ2 to reduce the mutual information by minimizing
LMI. We compute the mutual information with the traditional method and compare it with LMI:
I(X, Y ) =
Z
p(x, y) log p(y|x)
p(y) dxdy
≈
1
20000
20000
X
i=1
log p(yi|xi)
p(yi)
(19)
As is shown in Fig .9, the mutual information decreases as LMI descends. The advantage of such
bi-level objective to MI minimization in Eq.
11 is that it only requires samples instead of forms of
distribution.
Moreover, it needs no tractable prior distribution for variational approximation.
The
drawback is that it needs additional computation in the inner loop.
A.3
Algorithm
A.4
More results on graph interpretation
In Fig. 4, we show the distribution of absolute bias between the property of graphs and subgraphs. GIB
is able to generate such subgraphs with more similar properties to the original graphs.
In Fig. 5, we provide more results of four properties on graph interpretation.
A.5
More results on noisy graph classiﬁcation
We provide qualitative results on noisy graph classiﬁcation in Fig. 6.
3We use the toy dataset from https://github.com/mzgubic/MINE
10

## Page 11

Figure 3: We use the bi-level objective to minimize the mutual information of two distributions. The MI
is consistent with the loss as LMI declines.
Algorithm 1 Optimizing the graph information bottleneck.
Require: Graph G = {A, X}, graph label Y , inner-step T, outer-step N.
Ensure: Subgraph Gsub
1: function GIB(G = {A, X}, Y, T, N)
2:
θ ←θ0,
φ1 ←φ0
1
3:
for i = 0 →N do
4:
φ2 ←φ0
2
5:
for t = 0 →T do
6:
φt+1
2
←φt
2 + η1∇φt
2LMI
7:
end for
8:
θi+1 ←θi −η2∇θiL(θi, φi
1, φT
2 )
9:
φi+1
1
←φi
1 −η2∇φi
1L(θi, φi
1, φT
2 )
10:
end for
11:
Gsub ←g(G; θN)
12:
return Gsub
13: end function
11

## Page 12

Figure 4: The histgram of absolute bias between the property of graphs and subgraphs.
Figure 5: The molecules with its interpretation found by GIB. These subgraphs exhibit similar chemical
properties compared to the molecules on the left.
12

## Page 13

Figure 6: We show the blindly denoising results on permuted graphs. Each method operates on the
line-graphs and tries to recover the true topology by removing the redundant edges. Columns 4,5,6 shows
results obtained by diﬀerent methods, where “miss: m, wrong: n” means missing m edges and there are
n wrong edges in the output graph. GIB always recognizes more similar structure to the ground truth
(not provided in the training process) than other methods.
13
