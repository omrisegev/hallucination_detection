---
source_pdf: C:/Users/omris/TAU/hallucination_detection/papers/LOCA - LOcal Conformal Autoencoder.pdf
slug: loca-local-conformal-autoencoder
pages: 25
extracted_on: 2026-09-07
---

# LOCA - LOcal Conformal Autoencoder

## Page 1

LOCA: LOcal Conformal Autoencoder
for standardized data coordinates
Erez Peterfreund 1˚
Oﬁr Lindenbaum 2˚
Felix Dietrich 3
Tom Bertalan3
Matan Gavish1
Ioannis G. Kevrekidis3
Ronald R. Coifman2,:
1Hebrew University of Jerusalem; 2Yale University;
3Johns Hopkins University;
:Corresponding author. E-mail: ronald-coifman@yale.edu
˚ These authors contributed equally.
Abstract
We propose a deep-learning based method for obtaining standardized data coordi-
nates from scientiﬁc measurements.
Data observations are modeled as samples from
an unknown, non-linear deformation of an underlying Riemannian manifold, which is
parametrized by a few normalized latent variables. By leveraging a repeated measure-
ment sampling strategy, we present a method for learning an embedding in Rd that is
isometric to the latent variables of the manifold. These data coordinates, being invariant
under smooth changes of variables, enable matching between diﬀerent instrumental obser-
vations of the same phenomenon. Our embedding is obtained using a LOcal Conformal
Autoencoder (LOCA), an algorithm that constructs an embedding to rectify deformations
by using a local z-scoring procedure while preserving relevant geometric information. We
demonstrate the isometric embedding properties of LOCA on various model settings and
observe that it exhibits promising interpolation and extrapolation capabilities. Finally, we
apply LOCA to single-site Wi-Fi localization data, and to 3-dimensional curved surface
estimation based on a 2-dimensional projection.
1
Introduction
Reliable, standardized tools for analyzing complex measurements are crucial for science in the
data era. Experimental data often consist of multivariate observations of a physical object that
can be represented as an unknown Riemannian manifold. A key challenge in data analysis
involves converting the observations into a meaningful and, hopefully, intrinsic parametrization
of this manifold. For example, in astrophysics, one is interested in a representation that is
coherent with the material composition of stars based on measurable high dimensional spectro-
scopic data [1, 2]. This type of challenge has typically been studied under the broader umbrella
of dimensionality reduction and manifold learning, where numerous algorithmic solutions have
been proposed [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]. These methods rely on statistical or geomet-
rical assumptions and aim to reduce the dimension while preserving diﬀerent aﬃnities of the
observed high dimensional data.
1
arXiv:2004.07234v2  [cs.LG]  14 Jan 2021

## Page 2

In this paper, we focus on data obtained from several observation modalities measuring a
complex system. These observations are assumed to lie on a path-connected manifold which is
parameterized by a small number of latent variables. We assume that the measurements are
obtained via an unknown nonlinear measurement function observing the inaccessible manifold.
The task is then to invert the unknown measurement function, so as to ﬁnd a representation
that provides a standardized parametrization of the manifold. In general, this form of blind
inverse problem may not be feasible. Fortunately, in many cases, one can exploit a localized
measurement strategy, suggested in [13], to extract an embedding into internally standardized
(z-scored) latent variables.
Toward a useful formulation of our problem we note that, in numerous real-world scenarios,
it is possible to capture data using a localized “burst” sampling strategy [14, 15, 16, 17, 18, 19,
18, 20, 21, 22]. To motivate this type of “burst” sampling, we describe a toy experiment (see Fig.
1). Consider the task of recovering the geometry of a curved 2-dimensional homogeneous surface
in three dimensions using a laser beam, which heats the surface locally at several positions.
Here, a “burst” is realized through the brief local isotropic propagation of heat around each
laser impact location (each “data point”), which can be visualized as a local ellipse by a thermal
camera. Now, the task is to recover the curved geometry of the surface in three dimensions
using the collection of observed local 2-dimensional ellipses.
More generally, our strategy is realized by measuring such brief “bursts”, which are modeled
as local isotropic perturbations added to each state in the inaccessible latent manifold. The
“bursts” provide information on the local variability in a neighborhood of each data point.
Thus, they can be used to estimate the Jacobian (modulo an orthogonal transformation, as we
will discuss) of the unknown measurement function. The authors in [13] use such “bursts” and
suggest a scheme to recover a representation that is invariant to the unknown transformation.
Speciﬁcally, they use a local Mahalanobis metric, combined with eigenvectors of an Anisotropic
Laplacian, in order to extract a desired embedding.
Solutions such as [13] and extensions such as [23, 20, 24, 25, 26] can be used. There remain,
however, several challenges: (i) they require a dense sampling of the deformed manifold. (ii)
they deform the representation due to inherent boundary eﬀects, and (iii) they do not extend
easily to unseen samples. To overcome these limitations, we introduce the concept of a LOcal
Conformal Autoencoder (LOCA), a deep-learning based algorithm speciﬁcally suited to “burst”
measurements. LOCA is realized using an encoder-decoder pair, where the encoder attempts to
ﬁnd a mapping such that each “burst” is locally whitened (z-scored). By forcing reconstruction,
the decoder ensures that geometric information has not been lost. We have found LOCA to
be scalable, as well as easy to implement and parallelize using the existing deep learning open-
source codebase. We provide empirical evidence that the LOCA embedding is approximately
isometric to the latent manifold and extrapolates reliably to unseen samples. We discuss a
scheme to automatically tune the minimal embedding dimension of LOCA, and demonstrate
its precision in two real data problems.
The contributions in this paper are as follows. (i) We show that the localized sampling
strategy (our “bursts” at a given scale) generates a consistent Riemannian structure; under
certain conditions, it allows inverting the unknown measurement function (modulo a shift and
orthogonal transformation).
(ii) We present a two-step optimization scheme for learning a
parametric mapping, which is approximately an isometry of the latent manifold.
(iii) We
demonstrate the isometric properties of the extracted encoder on several examples. (iv) We
verify empirically that the extracted neural network has good interpolation and extrapolation
properties.
2

## Page 3

Figure 1: A motivating example for LOCA- learning the stereographic shape of a surface. We
consider a laser beam used to locally heat the surface at several positions. A thermal camera
measures the brief isotropic propagation of heat around each location. By scanning a thermal
image, we can identify the neighborhoods of each position, which we deﬁne as our “bursts”.
LOCA uses these “bursts” to invert the projection and recover a scaled version of the curved
surface.
2
Problem Settings
The burst measurement strategy.
Consider ﬁrst, for simplicity, the case where the latent domain for our system of interest is a
path-connected domain in a Euclidean space X “ Rd. We call X the latent space. Observations
of the system consist of samples captured by a measurement device given as a non-linear function
f : X Ñ Y, where Y is the ambient, or “measurement” space. Even if f is invertible, it is
generally not feasible to identify f ´1 without access to X. Here, we assume that (a) f is smooth
and injective; and that (b) multiple, slightly perturbed versions of the physical system point in
X give rise to multiple (slightly perturbed) measurements in Y. In this notation, by exploiting
a speciﬁc type of local perturbation, we develop a method to recover a standardized version X
from Y (up to an approximately isometric transformation which, for Euclidean spaces would
be a rigid transformation).
Consider N data points (“burst” centers), denoted x1, . . . , xN P Rd in the latent space.
Assume that all these points lie on a path-connected, d-dimensional sub-domain of X; we will
later discuss the restriction to Riemannian manifolds with a smaller dimension than the full
space. Importantly, we do not have direct access to these states. The states are pushed forward
to the ambient space via the unknown deformation which deﬁnes y1, . . . yN P RD. We do not
only observe the states y1, . . . , yN; we rather assume a speciﬁc perturbed sampling strategy.
For each xi, i “ 1, ..., N, we assume that the perturbed observed state is given as the random
variable
Y i “ fpxi ` Ziq P RD
i “ 1, . . . , N,
(1)
where Z1, . . . , Zn are i.i.d distributed by Ndp0, σ2Idq. Our sampling strategy relies on mea-
suring M perturbed realizations of Y i, which we denote as ypjq
i , j “ 1, ..., M. We assume that
σ ! 1, or alternatively, that σ is suﬃciently small such that the diﬀerential of f practically does
not change in a ball of radius σ around any point. Such suﬃciently small σ allows us to capture
the local neighborhoods of the states at this measurement scale on the latent manifold. Note
that alternative isotropic distributions, which satisfy this condition, could be used to model Z.
3

## Page 4

Let us explore the implications of this localized sampling strategy for learning a represen-
tation that is consistent with X. Speciﬁcally, our goal is to construct an embedding ρ that
maps the observations yi, so that the image of ρ ˝ f is isometric to X when σ is known. In our
Euclidean setting, such an isometric embedding should satisfy
}ρpyiq ´ ρpyjq}2 “ }xi ´ xj}2, for any i, j “ 1, . . . , N.
(2)
We note that if σ is not known, we will relax (2) by allowing a global scaling of the embedding.
This means that we are only looking for a representation that preserves the pairwise Euclidean
distances between the latent samples, rather than obtaining their actual values. More specif-
ically, a ρ that satisﬁes (2) is not unique, and is deﬁned up to an isometric transformation
of the data. We refer to representations which satisfy Eq. 2 up to errors smaller than σ as
“isometries”.
3
Related work
The problem of ﬁnding an isometric embedding was also studied in [27]. The paper proposes
an algorithm to embed a manifold with dimension d into a space of dimension s ě d. The
method in [27] is built upon [12] and uses a discrete version of the Laplace-Beltrami operator,
as in [7], to estimate metric of the desired embedding. To force the embedding to be isometric
to the observed samples Y, the authors propose a loss term to quantify the deviation between
the push-forward metric from the observed space, Y, to the embedding space compared with
the restricted Euclidean metric. The embedding is reﬁned by applying gradient descent to the
proposed loss. The approach successively approximates a Nash embedding with respect to the
observed space, Y, for which it is required that the manifold be densely sampled at all scales.
In this work, we use “bursts” to learn an embedding that corrects the deformation f and
isometrically represent the inaccessible manifold X. The idea of using “bursts”, or data neigh-
borhoods, to learn a standardized reparametrization of data was ﬁrst suggested in [13]. The
authors assume the data is obtained via some unknown nonlinear transformation of latent inde-
pendent variables. Locally, the distortion caused by the transformation is corrected by inverting
a Jacobian that is estimated from the covariances of the observed “bursts”. This allows the
authors to deﬁne a local Mahalanobis metric (which is aﬃne invariant). Then, this metric is
used to construct an anisotropic ”intrinsic Laplacian” operator. Finally, the eigenvectors of
this Laplacian provide the independent components and are used as a “canonical” embedding.
This framework was extended in several studies such as [24, 15, 23, 26].
The work of these authors can be improved in three directions. First, they require inverting
a covariance matrix in the ambient space. Second, they suﬀer from deformation on boundaries.
Third, they typically do not provide an embedding function that can be naturally extended over
the entire data domain and beyond; instead, they provide a speciﬁc mapping for the existing
training samples. This last direction means that to embed test data, methods such as [28, 14]
could be employed. The mapping approximations based on these methods are limited and
cannot extend further than a small neighborhood around each training point. Furthermore,
even though the provided embedding is unique, it is not isometric to the latent variables.
We present a method that alleviates these shortcomings, and empirically demonstrate that it
extracts a canonical representation that is isometric to the latent variables.
Perhaps the most related work to this study was recently presented by [19]. The authors
consider “bursts” to develop a method for ﬁnding an embedding that is isometric to the latent
variables. They build upon Isomap [29], and use two neural networks to reﬁne the Isomap
4

## Page 5

Figure 2: An illustration of the LOcal Conformal Autoencoder (LOCA). “E” stands for the
encoder ρ, and “D” for the decoder γ. The autoencoder receives a set of points along with
corresponding “neighborhoods”; each neighborhood is depicted as a dark oval point cloud (see
the top row in the ﬁgure).
On the bottom row, we zoom in onto a single “anchor” point
yi (green) along with its corresponding neighborhood Y i (bounded by a blue ellipsoid). The
encoder attempts to “whiten” each neighborhood in the embedding space, while the decoder
tries to reconstruct the input.
based embedding. The ﬁrst neural network is used in order to obtain a continuous model for
estimating the covariance CpY iq. The covariances are used for calculating local Mahalanobis
based distances, which are fed into Isomap to obtain an initial embedding. Next, they train
an additional neural network to correct the Isomap embedding so that the Euclidean distances
will approximate the local Mahalanobis distances. In this paper, we take a diﬀerent and, we
believe, more systematic/general approach by presenting a simple encoder-decoder pair (see Fig.
2) that is directly applicable to samples in the observed high dimensional space. Speciﬁcally,
our approach provides a parametric mapping that allows us to extend the embedding to new
unseen samples naturally. Furthermore, we learn the inverse mapping, which could be used to
generate new samples by interpolating in the latent space.
4
Deriving an Alternative Isometry Objective
Without access to samples from X the objective described in (2) does not provide any infor-
mation for extracting ρ. Here we reformulate this objective by utilizing the special stochastic
sampling scheme presented in Section 2 and relate it to the diﬀerential equation for the em-
bedding described in Lemma 1. We start by plugging the unknown measurement function into
(2); then, we can approximate its left hand side using a ﬁrst order Taylor expansion
}ρpyiq ´ ρpyjq}2
“
}ρ ˝ fpxiq ´ ρ ˝ fpxjq}2
«
}Jρ˝f pxiq pxj ´ xiq }2.
Hence, by neglecting higher order terms, we can deﬁne the following objective
Jρ˝f pxiqTJρ˝f pxiq “ Id, for i “ 1, . . . , N,
(3)
5

## Page 6

which allows us to evaluate the isometric property of ρ.
Now we want to relate the Jacobian in Eq. 3 to measurable properties of the observations
Y 1, . . . , Y N. Speciﬁcally, we can rely on the following Lemma to approximate the derivatives
of the unknown function f at each point x1, . . . , xN. The Lemma is proved in the Supporting
Information (A):
Lemma 1. Let g : X Ñ Z be a function, where X “ Rd and Z “ RD. Let x P X and σ P R`.
Deﬁne a random variable X „ Npx, σ2Idq. If the function satisﬁes g P C3 and is injective,
there exist a σ P R` such that the covariance of the transformed random variable Z “ gpXq is
related to the Jacobian of g at x via
JgpxqJgpxqT
“
1
σ2CpZq ` Opσ2q.
Moreover,
1
σ2CpZq ÝÑ
σÑ0 JgpxiqJgpxqT.
By setting g ” ρ ˝ f, this Lemma provides a relation between the Jacobian of ρ ˝ f and
the covariance in the embedding space. Speciﬁcally, this translates to a system of diﬀerential
equations for the Jacobian of an isometric (Nash) embedding
Jρ˝f pxiqJρ˝f pxiqT
“
1
σ2C pρpY iqq ` Opσ2q.
(4)
When D “ d, we can tie the approximation of objective (2) with (4) by
1
σ2C pρpY iqq
“
I, for any i “ 1, . . . , N.
(5)
Thus we can evaluate the embedding function at each point without gaining access to the latent
states of the system.
5
LOcal Conformal Autoencoder
We now introduce the Local COnformal Autoencoder (LOCA), with training Algorithm 1. Our
method is based on optimizing two loss terms; the ﬁrst is deﬁned based on (5) using what we
refer as a “whitening” loss
Lwhitepρq “ 1
N
N
ÿ
i“1
›››››
1
σ2 pC pρpY iqq ´ Id
›››››
2
F
,
(6)
where ρ is an embedding function, and pC pρpY iqq is the empirical covariance over a set of M
realizations ρ
´
yp1q
i
¯
, . . . , ρ
´
ypMq
i
¯
, where yp1q
i , . . . , ypMq
i
are realizations of the random variable
Y i.
As f is invertible on its domain, an embedding function ρ that approximates f ´1 should be
invertible as well. The invertibility of ρ means that there exists an inverse mapping γ : Rd ÝÑ
Y, such that yi “ γpρpyiqq for any i P rNs. This additional objective helps remove undesired
ambiguities (which may occur for insuﬃcient sampling). By imposing an invertibility property
6

## Page 7

Algorithm 1 LOCA: LOcal Conformal Autoencoder
Input: Observed clouds Y i,i=1,...,N.
Output:
θe
and θd
- the weights of the encoder ρ and decoder γ
neural net-
work.
1: for t “ 1, ..., T do
2:
Compute the whitening loss
Lwhite “ 1
N
N
ÿ
i“1
›› 1
σ2 pC pρpY iqq ´ Id
››2
F
3:
Update θe :“ θe ´ η∇θeLwhite
4:
Compute the reconstruction loss
Lrecon “
1
N ¨ M
N,M
ÿ
i,m“1
››ypmq
i
´ γ
´
ρ
´
ypmq
i
¯¯ ››2
2
5:
Update θe :“ θe ´ η∇θeLrecon and
θd :“ θd ´ η∇θdLrecon
6: end for
(a)
(b)
(c)
Figure 3:
Evaluating the isometric quality of the embedding (the setting is detailed in 6.1).
The points xi, i “ 1, ..., N “ 2000 in the latent space of the system of interest (a) are (b)
pushed forward to the measurements yi “ 1, ..., N by applying the nonlinear transformation
f 1(described in (8)). The color in both ﬁgures corresponds to the values of xr1s of the data.
We observe “bursts” around each sample (based on the burst model described in Section 2).
To illustrate this “burst” sampling scheme, we overlay the points with additional green samples
generated by bursts at 5 diﬀerent positions. (c) Euclidean distances between pairs of points in
the latent space plotted versus the corresponding Euclidean distance in the embedding space.
The corresponding distances for DM and for A-DM are also shown in color, scaled with a factor
which minimizes the stress (deﬁned in (9)).
on ρ, we eﬀectively regularize the solution of ρ away from noninvertible functions. To impose
7

## Page 8

invertibility, we deﬁne our second loss term, referred to as ”reconstruction” loss:
Lreconpρ, γq “
1
N ¨ M
ÿ
i,m
›››››ypmq
i
´ γ
´
ρ
´
ypmq
i
¯¯ ›››››
2
2
.
(7)
We suggest ﬁnding an isometric embedding based on an autoencoder, where ρ will be
deﬁned as the encoder and γ as the decoder. We construct solutions to (6) and (7) with a
neural network ansatz ρ “ hpLq
e
and γ “ hpLq
d
consisting of L layers each, such that
hpℓq
e pyq
“
σe
´
W pℓ´1q
e
hpℓ´1q
e
pyq ` bpℓ´1q¯
, ℓ“ 1, . . . , L,
hpℓq
d pzq
“
σd
´
W pℓ´1q
d
hpℓ´1q
d
pzq ` bpℓ´1q
d
¯
, ℓ“ 1, . . . , L,
where hp0q
e pyq “ y and hp0q
d pzq “ z. Here, W ℓ
e, bℓ
e and W ℓ
d, bℓ
d are the weights and biases at
layer ℓof the encoder and decoder, respectively. The functions σe, σd are nonlinear activations
applied individually to each input coordinate. As the activation function can have a limited
image, we recommend removing the non linear activation for ℓ“ L.
We propose to ﬁnd ρ and γ by alternating between a stochastic gradient descent on (6) and
(7). It is important to note that the main objective that we are trying to optimize is based
on (6), therefore (7) can be viewed as a regularization term. A pseudo-code of this procedure
appears in Algorithm 1. To prevent over-ﬁtting, we propose an early stopping procedure [30, 31]
by evaluating the loss terms on a validation set. In section 6, we demonstrate diﬀerent properties
of the proposed LOCA algorithm using various geometric example manifolds.
Note that functions that perfectly satisfy our objectives are not unique, i.e. for any solution
ρ we can deﬁne an equivalent solution ρ that will attain the same loss. Speciﬁcally, we can
deﬁne it by ρpyq “ Uy ` c for any y P Y, where U P Opdq and c P Rd.
To summarize: (i) We collect distorted neighborhoods of a ﬁxed size (σ) around data
points of the system of interest. (ii) We embed/encode the data in a low dimensional Euclidean
space so that these neighborhoods are standardized or z-scored. (iii) The embedding is decoded
back to the original measurements, to regularize the encoder. In Section 6, we demonstrate
that (iv) The encoder is invariant to the measurement modality (up to errors of Opσ2q, and
modulo an orthogonal transformation and shift). (v) The parametric form of the embedding
enables reliable interpolation and extrapolation.
6
Properties of LOCA
In this section, we evaluate the properties of the proposed embedding ρ by generating various
synthetic datasets. We compare the extracted embedding provided by LOCA (described in
Section 4) with the embeddings of alternative methods such as Diﬀusion Maps (DM) [7] and
Anisotropic Diﬀusion Maps (A-DM) [13] denoted as ψ and φ respectively.
Note that the
Diﬀusion Maps algorithm does not use the “burst” data, while the Anisotropic Diﬀusion Maps
uses it in order to construct a local Mahalanobis metric. We present the details of the exact
implementation for each of the methods in the supplementary information (see Section B).
6.1
LOCA creates an isometric embedding
We ﬁrst evaluate the isometric quality of the proposed embedding ρ with respect to the true
inaccessible structure of X.
Here, we follow the setting in [13], where the intrinsic latent
8

## Page 9

(a)
(b)
(c)
(d)
Figure 4: Evaluating out-of-sample performance of the encoder, detailed in Section 6.2. (a)
The latent space of interest X; our training region is bounded here by the black and green
frames. The interpolation region lies within the green frame, while the extrapolation regime
lies outside the black frame. (b) The observed space Y with corresponding regions of interest.
(c) The calibrated embedding ρ (using an orthogonal transformation and a shift) with corre-
sponding regions of interest. The color in these ﬁgures corresponds to the values of xr1s. Here,
we calibrated the embedding merely for visualisation purposes. (d) The Euclidean distances
between pairs of points in the latent space versus the corresponding Euclidean distance in the
embedding space.
coordinates are independent, speciﬁcally distributed by Ur0, 1s2. Based on this distribution, we
sample N “ 2000 anchor points xi along with “bursts” Xi. Each “burst” consists of M “ 200
points sampled independently from N2pxi, σ2I2q, where σ “ 0.01.
We now deﬁne the non linear transformation, f1 : R2 Ñ R2 (as in [13]) to the ambient space
by
f 1pxq
“
ˆ
xr0s ` xr1s3
´xr0s ` xr1s3
˙
,
(8)
for any x ” pxr0s, xr1sqJ P R2. In Fig. 3, we present measurements from X with the corre-
sponding measurements of Y, where Y “ fpXq. To illustrate the local deformation caused by
f 1, we overlay the samples with clouds around 5 diﬀerent positions (see green dots). Next,
we apply LOCA (described in Algorithm 1) to compute an embedding ρ that satisﬁes (6) and
(7). We evaluate the isometric quality of LOCA by comparing the pairwise Euclidean distances
in the embedding space ρ to the Euclidean distances in the latent space X. For comparison,
we apply DM and A-DM (that also uses the “bursts”) to Y and plot the pairwise Euclidean
distances in the embedding vs. the corresponding Euclidean distances in the latent space. Here,
we evaluate isometry up to a scaling, as DM and A-DM use eigenvectors (that are typically
normalized). The scaling is optimized to minimize the stress deﬁned by
Stresspgq
“
1
N
N
ÿ
i,j“1
`
Dxpxi, xjq ´ Dgpyi, yjq
˘2
(9)
where g is some embedding function from Y and Dgpyi, yjq “ }gpyiq ´ gpyjq}2. Speciﬁcally
the stress values for LOCA, and the scaled versions of DM and A-DM are 1.5 ¨ 10´5, 0.03 and
0.002, respectively. As evident from the stress values and from Fig. 3, LOCA provides an
embedding that is isometric to X (up to an orthogonal transformation and shift).
9

## Page 10

6.2
The encoder is observed to extend reliably to unseen samples
In the next experiment, we evaluate the out-of-sample extension capabilities of LOCA. The
experiment is based on the same nonlinear transformation described in Section 6.1. We sample
N “ 2, 000 points from a partial region of the latent representation X, speciﬁcally described
by r0, 1s2zr0.1, 0.9s2. In Fig. 4, we present the framed sampling regions along with the corre-
sponding observed framed regions in Y (see black and green frames 4(a) and 4(b)). To generate
the “bursts” we follow the setting presented in Section 6.1 and refer to them as our training
set. The test set is deﬁned by an additional 2 ¨ 104 samples generated as in Section 6.1 from
r´.025, 1.025s2 in X pushed forward by f 1.
In Fig.
4(c), we quantify the interpolation and extrapolation capabilities of LOCA by
presenting the extracted embedding along with the corrected frame. To further evaluate the
quality of this embedding we compare the pairwise distances in 4(d) (as described in Section
6.1).
This comparison (presented in Fig.
4(d)) supports the beneﬁts of using LOCA for
extending the representation to unseen samples. To stress it even more, the actual stress values
of LOCA in the interpolation region, on the frame and in the extrapolation region are all
approximately 10´4.
(a)
(b)
(c)
(d)
Figure 5: Evaluating the out-of-sample reconstruction capabilities of LOCA. Here, we attempt
to generate new point in the ambient space by performing linear interpolation in the embedding
space. A description of the linear interpolation appears in 6.3. (a) The inaccessible latent space,
the points surrounding the green frame are the training samples. Interpolation is performed
horizontally and vertically between points on the green frame, see for example the 4 colored
lines. (b) The pushed forward data from (a) based on the non linear function f 1 described in (8).
(c) Recovered calibrated embedding ρ. The training samples in the frame and green boarder
are embedded using LOCA. Within the embedded green boarder we perform an additional
linear interpolation, using the same corresponding pairs as were used in the latent space. For
example, see the 4 horizontal and vertical colored lines. (d) The pushed forward data from
(c) by the decoder (γ) learned by LOCA. This experiment demonstrates that LOCA learns a
decoding function that is consistent with the unknown transformation, even in a regime that
is not covered by training samples.
6.3
The decoder is observed to extend reliably to unseen samples
In this experiment we evaluate the out-of-sample capabilities of LOCA’s decoder. While in
6.2 we trained LOCA and evaluated the quality of the encoder on unseen data, here we focus
on the performance of the decoder. Speciﬁcally, we apply the decoder to unseen samples from
10

## Page 11

the embedding space.
Each unseen sample in the embedding space is created using linear
interpolation. We now provide the exact details of this evaluation.
For this interpolation experiment, we use the same LOCA model trained in 6.2 on the framed
data. We further generate N “ 400 points in the interior boundary of the frame, represented by
the green dots in Fig. 5. Next, we perform linear interpolation between horizontal and vertical
pairs, see for example the colored lines in Fig. 5(a). The data is then pushed forward using
the nonlinear transformation described in (8), as shown in Fig. 5(b). We embed the training
samples along with the green frame using LOCA; a calibrated version of the embedding space
is shown in Fig. 5(c) (up to a shift and an orthogonal transformation). Then, we perform an
additional interpolation in the embedding space using the same corresponding pairs as were
used in the latent space (see Fig.
5(a)).
Finally, we apply the decoder to the embedding
of the training samples and to the newly interpolated samples, these are presented in Fig.
5(d).
As evident in this ﬁgure, the reconstructed points faithfully capture the mushroom
shaped manifold. The mean squared error between the push-forward interpolated points and
the decoded interpolated points is 2.3 ¨ 10´4, with a standard deviation of 2.4 ¨ 10´4. This
experiment demonstrates that LOCA may be also used as a generative model, by reconstructing
new points generated using interpolation in the embedding space.
6.4
LOCA on a curved manifold
Here, we examine a more challenging conﬁguration, generalizing our original Euclidean problem
setting. The latent space is now taken to be a k´dimensional manifold that resides in Rd,
where d ą k and d is the minimal dimension required to embed the manifold in a Euclidean
space isometrically. Interestingly, we consider an observation process such that the observation
dimension, D, is smaller than d. To clarify, this means that the measurement process can
involve projections to a lower dimension.
We consider a manifold that covers three quarters of a 2-dimensional unit sphere in R3,
where the training points admit the following form
x “
¨
˝
sinpαq cospβq
sinpαq sinpβq
cospαq
˛
‚
β P r0, 2πq, α P rπ{3, πs.
(10)
The manifold is embedded in R3 but has an intrinsic dimension of 2. This requires us to
revisit our deﬁnition of “bursts”, discussed in (1). Speciﬁcally, we assume that the bursts are
conﬁned to the manifold. Here, we approximate this constraint in the form of random variables
Zi obtained using a local isotropic Gaussian with a two dimensional covariance σ2I2, deﬁned
on the tangent plane to the point.
We consider N “ 491 states of the system xi, i “ 1, .., N, which are generated on a uniform
grid using the “Fibonacci Sphere” sampling scheme [32] for points with α P rπ{3, 5π{6s. We
deﬁne each “burst” Xi using M “ 400 points sampled from our two-dimensional isotropic
Gaussian deﬁned by the tangent plane around xi with σ “ 0.01. Now, in order to create the
observed samples y we apply the stereographic projection to x by projecting each point from
X onto a two dimensional space deﬁned by:
y “
˜
xr1s
1´xr3s
xr2s
1´xr3s
¸
.
(11)
The transformation can be thought of as a projection onto the plane R2 ˆ t1u; an illustration
of the stereographic projection appears in Fig. 6(a). The training “bursts” in the latent space
and the observed space appear in Figs. 6(b) and 6(c), respectively.
11

## Page 12

(a)
(b)
(c)
(d)
(e)
Figure 6: The stereographic projection experiment (see description in Section 6.4). (a): A
schematic illustration of the stereographic projection generating the data. (b): The original
latent representation of the “bursts” employed. (c): The 2-dimensional observations of the
“bursts” created using the stereographic projection. The plot contains only the training data,
meaning points that satisfy α P pπ{3, 5π{6q, leaving a “hole” at the south pole. (d): The 3-
dimensional embedding of these training data, with the missing lower cap (α P pπ{3, πs). The
color represents the value of α of each point as deﬁned in (10). The colors used in (b)- (d)
correspond to the spherical angle α deﬁned in (10). (e) The Euclidean distances between pairs
of points in the original, 3-dimensional latent space versus the corresponding Euclidean distance
in the embedding space. Here we compare distances based on the training region (frame) as
well as the unseen test region where α P pπ{6, πs (interpolation).
We apply DM, Anisotropic DM, and LOCA to embed the data in a 3-dimensional space.
The diﬀerence between the pairwise Euclidean distances (see description in Section 6.1) in each
embedding space and the original Euclidean distances along with the extracted embeddings are
described in the Supplementary Information. The stress values for LOCA, and the scaled DM
and A-DM on the training data are 10´3, 0.18 and 6¨10´3, respectively. In order to examine the
interpolation capabilities of LOCA, we generate 55 points using the ”Fibonacci Sphere” that
satisﬁes α P p5π{6, πs. Using the trained model of LOCA we embed these data and get that the
stress value is 10´4 . Fig. 6 demonstrates that LOCA can well approximate an isometry even if
the dimension of the observations is lower than the minimal embedding dimension needed for
the isometry, i.e. k ą D.
12

## Page 13

7
Applications
7.1
Flattening a curved surface
Our ﬁrst application is motivated by [33], in which the authors propose a method for estimating
the 3-dimensional deformation of a 2-dimensional object. They focus on the task of autonomous
robotic manipulation of deformable objects. Their method uses a stream of images from a RGB-
D camera and aligns them to a reference shape to estimate the deformation.
We explore the applicability of LOCA for the task of estimating a deformation based on
a 2-dimensional projection of an object, without using any depth feature. We print a black
square-shaped grid of N “ 2500 points of interest; at each location, we generate a “burst” with
M “ 50 samples drawn from a Gaussian with σ “ 0.01. We manually squeeze the printed
square and photograph the deformed object from above. The image of the original squared
object along with a 2-dimensional snapshot of the deformed object appears in Fig. 7(a). This
experiment complements our motivating example presented in Fig. 1.
To deﬁne the anchor points yi along with corresponding “bursts”, we ﬁrst identify the
locations of all points by applying a simple threshold ﬁlter to the image. Then, we identify
the “bursts” by applying the Density-Based Spatial Clustering of Applications with Noise
(DBSCAN) [34]. In Fig. 7(a) we present the identiﬁed groups of points (black). Note that some
“bursts” are lost in this process, as there is nearly no gap between them in the deformed shape.
Here, the parameter σ2 for the whitening loss (6) is estimated using the median of the ﬁrst
eigenvalue of the “bursts” covariances. We apply LOCA and extract the embedding ρ. In Fig.
7(b), we present a calibrated version (scaled rigid transformation) of the embedding, ˜ρ, overlaid
on the latent representation. The transformation is found by minimizing the mean squared
error between the underlying representation and the extracted embedding of the 4 corners of
the square. This experiment demonstrates that LOCA corrects the unknown deformation based
on the estimated “bursts”.
7.2
Application to Wi-Fi localization
Here, we evaluate LOCA for the task of geographical localization based on mobile devices. The
localization problem involves estimating the geographic location of a receiver device based on
signals sent from multiple Wi-Fi transmitters. This problem has been addressed by modeling
the strength of the signal in time and space, or based on ﬁngerprints of the signal learned in
a supervised setting [35, 36]. We address the problem in an unsupervised fashion by applying
the proposed LOCA algorithm without employing any physical model.
The experiment is performed by simulating the signal strength of L “ 17 Wi-Fi transmitters
at multiple locations across a model of a room, where each transmitter uses a separate channel.
The room is modeled based on a simpliﬁed ﬂoor plan of the fourth ﬂoor of MIT’s Ray and
Maria Stata Center. We refer to the two-dimensional representation of the room as X Ă R2; a
schematic of the ﬂoor plan with 600 ˆ 1000 pixels appears in Fig. 8 (black line). The L “ 17
Wi-Fi transmitters are randomly located across the ﬂoor plan; we denote each of these locations
by tℓP R2, for any ℓP t1, ..., 17u. Next, we sample xi, i “ 1, ..., N, using N “ 4, 000 anchor
points distributed uniformly over X and deﬁne the amplitude of each measured Wi-Fi signal
using a radial basis function (RBF). The RBF decay is monotonic in the distance between the
transmitter and the measurement location, so that the amplitude at point xi of the signal of
transmitter ℓis yi,ℓ“ expp´}xi ´ tℓ}2
2{ϵ2q, where ϵ “ 600 pixels. Here the “bursts” will be
deﬁned by a circle of M “ 6 receivers equally spaced at a radius of r “ 0.5 pixels around each
anchor point xi: these 6 receivers model a circular sensor array as the measurement device.
13

## Page 14

(a)
(b)
Figure 7: A LOCA embedding can ﬂatten a deformed object using estimated “bursts”. (a)
The input training samples used by LOCA (black) and the reconstructed points (red). (b) The
calibrated embedding (using an orthogonal transformation and a shift) of the deformed object
using LOCA (red) and the underlying representation of points (black). As in the synthetic
examples, we use calibration only for visualization purposes. Here, LOCA manages to correct
the deformation of the local “bursts”, and thus learns a function that approximately uncovers
the latent structure of the object.
Next, we apply LOCA and embed the observed vectors of multi-channel amplitudes into
a 2-dimensional space.
To demonstrate the performance of LOCA we calibrate the LOCA
embedding to the ground truth ﬂoor plan using a shift and scaled orthogonal transformation,
as done in 7.1 but using all the training data. In Fig. 8 we present the scaled, calibrated
two-dimensional embedding ˜ρ with the locations of the transmitters and anchor points.
8
Discussion
We propose a method that extracts canonical data coordinates from scientiﬁc measurements.
These approximate an embedding that is isometric to the latent manifold. We are assuming
a speciﬁc, broadly applicable stochastic sampling strategy, and our proposed method corrects
for unknown measurement device deformations. Our method constructs a representation that
”whitens” (namely, changes to multivariate z-scores) groups of local neighborhoods, which
we call “bursts”.
We impose additional constraints to patch together the locally whitened
neighborhoods, ensuring a smooth global structure. Finally, the method is implemented using a
neural network architecture, namely an encoder-decoder pair, which we name LOcal Conformal
Autoencoder (LOCA).
The method can be summarized as follows. (i) We collect distorted neighborhoods of a
ﬁxed size of data samples.(ii) We embed/encode the data in the lowest dimensional Euclidean
space so that these neighborhoods are standardized or z-scored.(iii) The data is decoded from
embedding space to original measurements, enabling interpolation and extrapolation.
(iv)
LOCA is invariant to the measurement modality (approximately to second order, and modulo
14

## Page 15

Figure 8: Application of LOCA to Wi-Fi localization. We use a ﬂoor plan model based on the
fourth ﬂoor of MIT’s Ray and Maria Stata Center. The edges of the ground truth model appear
in black. We simulate L “ 17 Wi-Fi access points (transmitters), which are presented as black
crosses. We use N “ 4000 locations depicted as red dots with corresponding M “ 6 “burst
samples” around them (modeling a circular antenna array).
To demonstrate that LOCA’s
embedding is coherent with the latent representation, we calibrate the embedding to the true
ﬂoor plan, see blue dots and green line.
a rigid transformation). (v) Under scaling consistency for samples drawn from a Riemannian
manifold, the encoder can approximate an isometric embedding of the manifold.
From an implementation perspective, our method is simpler than existing manifold learning
methods, which typically require an eigen-decomposition of an N-by-N matrix (N being the
number of samples). Indeed, existing implementations of deep neural networks enable a single
developer to produce fast, reliable, GPU-based implementations of LOCA.
We provided solid empirical evidence that, if the deformation is invertible, then LOCA
extracts an embedding that is isometric to the latent variables.
Moreover, LOCA exhibits
intriguing, indeed promising interpolation and extrapolation capabilities.
To motivate the
beneﬁts of using LOCA, we used two potential applications. First, we apply a 3-dimensional
deformation to a printed object and demonstrate that LOCA manages to invert the deformation
without using any assumptions on the object’s structure. Finally, using Wi-Fi generated signals
from multiple locations, we show that the LOCA embedding can be quantititatively correlated
with the true locations of the received signals.
Our method relies on the “bursts” measurement model. As shown in Lemma 1, the co-
variances of the bursts can be used to estimate the Jacobian of the unknown measurement
function. Alternatively, we can replace this estimation with any other type of measurement
strategy informative enough to estimate the local Jacobian of the measurement function.
15

## Page 16

Acknowledgements
This work was partially supported by the DARPA PAI program (Agreement No. HR00111890032,
Dr. T. Senator). This material is based upon work supported by, or in part by, the U. S.
Army Research Laboratory and the U. S. Army Research Oﬃce under contract/grant num-
ber W911NF1710306. E.P. has been partially supported by the Blavatnik Interdisciplinary
Research Center (ICRC), the Federmann Research Center (Hebrew University) and Israeli Sci-
ence Foundation research grant no. 1523/16.
References
[1] S. Alam, F. D. Albareti, C. A. Prieto, F. Anders, S. F. Anderson, T. Anderton, B. H.
Andrews, E. Armengaud, ´E. Aubourg, S. Bailey, et al., “The eleventh and twelfth data
releases of the sloan digital sky survey: ﬁnal data from sdss-iii,” The Astrophysical Journal
Supplement Series, vol. 219, no. 1, p. 12, 2015.
[2] V. F. Calderon and A. A. Berlind, “Prediction of galaxy halo masses in sdss dr7 via a
machine learning approach,” Monthly Notices of the Royal Astronomical Society, vol. 490,
no. 2, pp. 2367–2379, 2019.
[3] H.-P. Deutsch, “Principal component analysis,” in Derivatives and Internal Models,
pp. 539–547, Springer, 2002.
[4] J. B. Kruskal, “Nonmetric multidimensional scaling: a numerical method,” Psychometrika,
vol. 29, no. 2, pp. 115–129, 1964.
[5] J. B. Tenenbaum, V. de Silva, and J. C. Langford, “A global geometric framework for
nonlinear dimensionality reduction,” Science, vol. 290, no. 5500, pp. 2319–2323, 2000.
[6] M. Belkin and P. Niyogi, “Laplacian eigenmaps for dimensionality reduction and data
representation,” Neural computation, vol. 15, no. 6, pp. 1373–1396, 2003.
[7] R. R. Coifman and S. Lafon, “Diﬀusion maps,” Applied and computational harmonic anal-
ysis, vol. 21, no. 1, pp. 5–30, 2006.
[8] S. T. Roweis and L. K. Sau, “Nonlinear dimensionality reduction by local linear embed-
ding,” Science, vol. 290.5500, pp. 2323–2326, 200.
[9] P. Baldi, “Autoencoders, unsupervised learning, and deep architectures,” in Proceedings
of ICML workshop on unsupervised and transfer learning, pp. 37–49, 2012.
[10] L. v. d. Maaten and G. Hinton, “Visualizing data using t-sne,” Journal of machine learning
research, vol. 9, no. Nov, pp. 2579–2605, 2008.
[11] L. McInnes, J. Healy, N. Saul, and L. Großberger, “Umap: uniform manifold approx-
imation and projection,” The Journal of Open Source Software, vol. 3, no. 29, p. 861,
2018.
[12] D. Perraul-Joncas and M. Meilˆa, “Non-linear dimensionality reduction: Riemannian metric
estimation and the problem of geometric discovery,” arXiv preprint arXiv:1305.7255, 2013.
16

## Page 17

[13] A. Singer and R. R. Coifman, “Non-linear independent component analysis with diﬀusion
maps,” Applied and Computational Harmonic Analysis, vol. 25, pp. 226–239, 9 2008.
[14] N. Rabin and R. R. Coifman, “Heterogeneous datasets representation and learning using
diﬀusion maps and laplacian pyramids,” in Proceedings of the 2012 SIAM International
Conference on Data Mining, pp. 189–199, SIAM, 2012.
[15] R. Talmon, I. Cohen, and S. Gannot, “Supervised source localization using diﬀusion ker-
nels,” in 2011 IEEE workshop on applications of signal processing to audio and acoustics
(WASPAA), pp. 245–248, IEEE, 2011.
[16] D. Kushnir, A. Haddad, and R. R. Coifman, “Anisotropic diﬀusion on sub-manifolds
with application to earth structure classiﬁcation,” Applied and Computational Harmonic
Analysis, vol. 32, no. 2, pp. 280–294, 2012.
[17] H.-t. Wu, R. Talmon, and Y.-L. Lo, “Assess sleep stage by modern signal processing
techniques,” IEEE Transactions on Biomedical Engineering, vol. 62, no. 4, pp. 1159–1168,
2014.
[18] X. Wang, H. Mao, H. Hu, and Z. Zhang, “Crack localization in hydraulic turbine blades
based on kernel independent component analysis and wavelet neural network,” Interna-
tional Journal of Computational Intelligence Systems, vol. 6, no. 6, pp. 1116–1124, 2013.
[19] A. Schwartz and R. Talmon, “Intrinsic isometric manifold learning with application to
localization,” SIAM Journal on Imaging Sciences, vol. 12, no. 3, pp. 1347–1391, 2019.
[20] A. Singer, R. Erban, I. G. Kevrekidis, and R. R. Coifman, “Detecting intrinsic slow vari-
ables in stochastic dynamical systems by anisotropic diﬀusion maps,” Proceedings of the
National Academy of Sciences, vol. 106, no. 38, pp. 16090–16095, 2009.
[21] S. A. Shevchik, B. Meylan, G. Violakis, and K. Wasmer, “3d reconstruction of cracks prop-
agation in mechanical workpieces analyzing non-stationary acoustic mixtures,” Mechanical
Systems and Signal Processing, vol. 119, pp. 55–64, 2019.
[22] R. Talmon and R. R. Coifman, “Diﬀerential stochastic sensing: Intrinsic modeling of
random time series with applications to nonlinear tracking,” Proc. Nat. Acad. Sci, 2012.
[23] R. Talmon and R. R. Coifman, “Empirical intrinsic geometry for nonlinear modeling and
time series ﬁltering,” Proceedings of the National Academy of Sciences, vol. 110, no. 31,
pp. 12535–12540, 2013.
[24] C. J. Dsilva, R. Talmon, N. Rabin, R. R. Coifman, and I. G. Kevrekidis, “Nonlinear intrin-
sic variables and state reconstruction in multiscale simulations,” The Journal of chemical
physics, vol. 139, no. 18, p. 11B608 1, 2013.
[25] F. P. Kemeth, S. W. Haugland, F. Dietrich, T. Bertalan, K. H¨ohlein, Q. Li, E. M. Bollt,
R. Talmon, K. Krischer, and I. G. Kevrekidis, “An emergent space for distributed data with
hidden internal order through manifold learning,” IEEE Access, vol. 6, pp. 77402–77413,
2018.
[26] M. Salhov, O. Lindenbaum, Y. Aizenbud, A. Silberschatz, Y. Shkolnisky, and A. Averbuch,
“Multi-view kernel consensus for data analysis,” Applied and Computational Harmonic
Analysis, 2019.
17

## Page 18

[27] J. McQueen, M. Meila, and D. Joncas, “Nearly isometric embedding by relaxation,” in
Advances in Neural Information Processing Systems, pp. 2631–2639, 2016.
[28] R. R. Coifman and S. Lafon, “Geometric harmonics: a novel tool for multiscale out-of-
sample extension of empirical functions,” Applied and Computational Harmonic Analysis,
vol. 21, no. 1, pp. 31–52, 2006.
[29] J. B. Tenenbaum, V. De Silva, and J. C. Langford, “A global geometric framework for
nonlinear dimensionality reduction,” science, vol. 290, no. 5500, pp. 2319–2323, 2000.
[30] M. Li, M. Soltanolkotabi, and S. Oymak, “Gradient descent with early stopping is
provably robust to label noise for overparameterized neural networks,” arXiv preprint
arXiv:1903.11680, 2019.
[31] R. Basri, D. Jacobs, Y. Kasten, and S. Kritchman, “The convergence rate of neural net-
works for learned functions of diﬀerent frequencies,” arXiv preprint arXiv:1906.00425,
2019.
[32] R. Swinbank and R. James Purser, “Fibonacci grids: A novel approach to global mod-
elling,” Quarterly Journal of the Royal Meteorological Society: A journal of the atmospheric
sciences, applied meteorology and physical oceanography, vol. 132, no. 619, pp. 1769–1793,
2006.
[33] T. Han, X. Zhao, P. Sun, and J. Pan, “Robust shape estimation for 3d deformable object
manipulation,” arXiv preprint arXiv:1809.09802, 2018.
[34] E. Schubert, J. Sander, M. Ester, H. P. Kriegel, and X. Xu, “Dbscan revisited, revisited:
why and how you should (still) use dbscan,” ACM Transactions on Database Systems
(TODS), vol. 42, no. 3, p. 19, 2017.
[35] A. Jaﬀe and M. Wax, “Single-site localization via maximum discrimination multipath
ﬁngerprinting,” IEEE Transactions on Signal Processing, vol. 62, no. 7, pp. 1718–1728,
2014.
[36] N. Alikhani, S. Amirinanloo, V. Moghtadaiee, and S. A. Ghorashi, “Fast ﬁngerprinting
based indoor localization by wi-ﬁsignals,” in 2017 7th International Conference on Com-
puter and Knowledge Engineering (ICCKE), pp. 241–246, IEEE, 2017.
[37] Y. Keller, R. R. Coifman, S. Lafon, and S. W. Zucker, “Audio-visual group recognition
using diﬀusion maps,” IEEE Transactions on Signal Processing, vol. 58, no. 1, pp. 403–413,
2009.
[38] L. Zelnik-Manor and P. Perona, “Self-tuning spectral clustering,” in Advances in neural
information processing systems, pp. 1601–1608, 2005.
[39] O. Lindenbaum, M. Salhov, A. Yeredor, and A. Averbuch, “Kernel scaling for manifold
learning and classiﬁcation,” arXiv preprint arXiv:1707.01093, 2017.
18

## Page 19

Appendix A
Proof of Lemma
Proof of lemma 1. For a suﬃciently small σ and any x such that }x ´ xi}2
2 “ Opσ2q we can
express gpxq by
gpxq “ gpxiq ` Jgpxiqpx ´ xiq ` Op}x ´ xi}2
2q;
here we use the smoothness of g. Hence, we can express the covariance of the random variable
Y i “ gpXiq by
CpY iq
“
ErpY i ´ ErY isq pY i ´ ErY isqTs
“
Er
`
Y i ´ gpxiq ` Opσ2q
˘
¨
`
Y i ´ gpxiq ` Opσ2q
˘Ts
“
J gpxiqErpx ´ xiqpx ´ xiqTsJ T
g pxiq ` Opσ4q
“
σ2J gpxiqJ T
g pxiq ` Opσ4q,
where
ErY is
“
ErgpXiqs
“
Ergpxiq ` JgpxiqpXi ´ xiq ` p}Xi ´ xi}2
2qs
“
gpxiq ` Opσ2q.
Appendix B
Methods Description
Here we provide a description and implementation details for Diﬀusion maps (DM) [7] and
Anisotropic Diﬀusion maps (A-DM) [13].
B.1
Diﬀusion maps
Diﬀusion maps (DM) [7] is a kernel based method for non linear dimensionality reduction.
The method relies on a stochastic matrix built using a kernel K : MY ˆ MY Ñ R. The
stochastic matrix can be viewed as a ﬁctitious random walk on the graph of the data. The
reduced representation is obtained via an eigendecomposition of the stochastic matrix. The
DM construction is summarized in the following steps:
1. Deﬁne a kernel function K : Y ˆ Y ÝÑ R, such that K P RNˆN with elements Ki,j “
Kpyi, yjq, where K is symmetric, positive semi-deﬁnite and non-negative. Here, we focus
on the common Radial Basis kernel deﬁned for any yi and yj as
Kpyi, yjq ﬁKi,j “ exp
ˆ
´||yi ´ yj||2
2ϵ
˙
, i, j P t1 . . . Nu,
(12)
where ϵ is a kernel bandwidth (see more details below).
2. Row normalize K
P ﬁD´1K P RNˆN,
(13)
19

## Page 20

where the diagonal matrix D P RNˆN is deﬁned as Di,i “ ř
j Ki,j. P can be interpreted
as the matrix of transition probabilities of a Markov chain on Y , such that rpP qtsi,j ﬁ
ptpyi, yjq (where t is an integer power) describes the implied probability of transition
from point yi to point yj in t steps.
3. Deﬁne the embedding for the dataset Y by
Ψt : pyiq : xi ÞÝÑ
“
λt
1ψ1piq, λt
2ψ2piq, λt
3ψ3piq, ..., λt
dψdpiq
‰T P Rd,
(14)
where λi and ψi are the i-th eigenvalue and right eigenvector of the matrix P .
It is important to properly tune the kernel scale/bandwidth ϵ, which determines the scale
of connectivity of the kernel K. Several studies have suggested methods for optimizing ϵ in
DM (e.g. [20, 37, 38, 39]). Here, we use the max-min approach initially suggested in [37] where
the scale is set to
ϵMaxMin “ max
j rmin
i,i‰jp||yi ´ yj||2qs, i, j “ 1, ...N.
(15)
The max-min aims for a scale that ensures that all points are connected to at least one other
point.
B.2
Anisotropic Diﬀusion maps
The Anisotropic Diﬀusion maps (A-DM) proposed in [13] eﬀectively replace the Euclidean
distance in (12) by a (joint) local Mahalanobis distance.
This local Mahalanobis distance
between observed anchor points yi and yj is computed using the observed “short bursts” by
}yi ´ yj}2
M ﬁ1
2pyi ´ yjqTrC:
i ` C:
jspyi ´ yjq,
(16)
where C:
i is the generalized inverse of the sample covariance of the i-th observation burst. This
joint local Mahalanobis distance is used to compute the Anisotropic diﬀusion kernel
rKpyi, yjq ﬁrKi,j “ exp
ˆ
´||yi ´ yj||2
M
2ϵ
˙
, i, j P t1 . . . Nu;
(17)
the scaling parameter is again optimized using (15) but now based on the Mahalanobis metric.
Next, we follow steps (2) and (3) in DM (see Section B.1) and compute the A-DM embedding
φ using the right eigenvectors of the normalized kernel.
Appendix C
Numerical Experiment Setting.
Here we describe the implementation details of each numerical experiment undertaken. The
architectures we used are based on fully connected layers, with an activation function after each
layer, except for the last two layers.
• First experiment details (3)
– Neural net architecture-
Encoder- Neurons in each layer: [50,50,2,2], Activation function: tanh.
Decoder- Neurons in each layer: [50,50,2,2], Activation function: tanh.
20

## Page 21

– Data- amount clouds: N “ 2000, cloud size: M “ 200, noise standard noise: σ “
1e ´ 2.
– Neural net training- batch size: 200 clouds, amount training clouds: 1800, amount
validation clouds: 200.
• Second and third experiment details (4,5)
– Neural net architecture-
Encoder- Neurons in each layer: [50,50,2,2], Activation function: tanh.
Decoder- Neurons in each layer: [50,50,2,2], Activation function: tanh.
– Data- amount clouds: N “ 2000, cloud size: M “ 200, noise standard noise: σ “
1e ´ 2.
– Neural net training- batch size: 200 clouds, amount training clouds: 1800, amount
validation clouds: 200.
• Forth experiment details (6)
– Neural net architecture-
Encoder- Neurons in each layer: [100,100,3,3], Activation function: tanh.
Decoder- Neurons in each layer: [100,100,2,2], Activation function: leaky relu.
– Data- amount clouds: N “ 546, cloud size: M “ 400, noise standard noise: σ “
1e ´ 2.
The points on the sphere were generated using the ”Fibonacci Sphere” mechanism
with 800 points. The points that did not satisfy constraints were left out.
– Neural net training- batch size: 50 clouds, amount training clouds: 491, amount
validation clouds: 55.
– Generation of a cloud around a given point on the unit sphere- Let pα, βq be
its polar representation.
We sample a cloud of points in the polar space using
N2ppπ{2, 0q, σ2I2q. Next, we ﬁnd some orthogonal transformation that maps pπ{2, 0q
to pα, βq in the 3-dimensional Cartesian space, and apply it to each sampled point.
• Flattening a curved surface experiment details(7.1)
– Neural net architecture-
Encoder- Neurons in each layer: [200,200,2,2], Activation function: tanh.
Decoder- Neurons in each layer: [200,200,2,2], Activation function: leaky relu.
– Data- amount clouds: N “ 2500, cloud size: M “ 60.
– Neural net training- batch size: 250 clouds, amount training clouds: 2250, amount
validation clouds: 250
• Wi-Fi localization experiment details (7.2)
– Neural net architecture-
Encoder- Neurons in each layer: [200,200,3,3], Activation function: tanh.
Decoder- Neurons in each layer: [200,200,2,2], Activation function: leaky relu.
– Data- amount clouds: N “ 4000, cloud size: M “ 6.
– Neural net training- batch size: 200 clouds, amount training clouds: 3600, amount
validation clouds: 400.
21

## Page 22

– Generation of a cloud around a given point - sample the circle every π{3 starting at
0.
The LOCA model was trained using an ADAM optimizer, that minimized at each epoch
one of the two loss (6),(7). It was trained using 90% of the given clouds, while the rest was
deﬁned as a validation set. The early stopping mechanism was implemented by evaluating the
sum of the two losses every 100 epochs. It saves the minimal value and the weights of the model
that achieved this loss. The neural net terminates its training when the minimal loss was not
changed in the last 2000 epochs and loads the saved weights. By following this description we
trained LOCA with the sequence of learning rates 10´3, then we ﬁne tuned it with learning
3 ¨ 10´4, and ﬁnished by training it with the learning rate 10´4.
22

## Page 23

(a)
(b)
(c)
(d)
Figure 9: Evaluating the isometric property of the proposed embedding (as described in Section
6.1). (a)- The latent representation of the data. (b)- The scaled calibrated embedding of DM.
(c) The scaled calibrated embedding of A-DM. (d) The calibrated embedding of LOCA. The
color in both ﬁgures correspond to the values of xr1s of the data.
23

## Page 24

(a)
(b)
(c)
(d)
Figure 10: Evaluating three dimensional embedding of the stereographic projection (described
in (10)). We compare the latent representation of the data (a) with its embedding based of
DM (b), A-DM (c) and LOCA (d). The colors used in (b),(c) and (d) is α deﬁned in (10)
24

## Page 25

(a)
(b)
Figure 11: A demonstration of the minimal embedding dimension estimation (E), based on
the procedure described in Section 6.4. 11(a): Loss values on the validation set for embedding
dimensions in the range 1, . . . , 5. 11(b): Plot of suggested quantity for estimating the minimal
embedding dimension. This plot suggests that 3 coordinates are suﬃcient to represent the data
using LOCA.
25
