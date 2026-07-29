---
source_pdf: papers/Einstein from Noise - Statistical Analysis.pdf
slug: einstein-from-noise-statistical-analysis
pages: 78
extracted_on: 2026-07-29
---

# Einstein from Noise - Statistical Analysis

## Page 1

Einstein from Noise: Statistical Analysis
Amnon Balanov∗1, Wasim Huleihel1, and Tamir Bendory1
1School of Electrical and Computer Engineering, Tel Aviv University, Tel Aviv 69978, Israel
March 11, 2026
Abstract
“Einstein from noise” (EfN) is a prominent example of the model bias phenomenon,
where systematic errors in the statistical model lead to spurious but consistent esti-
mates. In the EfN experiment, one falsely believes that a set of observations contains
noisy, shifted copies of a template signal (e.g., an Einstein image), whereas in reality,
it contains only pure noise observations. To estimate the signal, the observations are
first aligned with the template using cross-correlation and then averaged. Although
the observations contain nothing but noise, it was recognized early on that this process
produces a signal that resembles the template signal! This model bias pitfall was at
the heart of a central scientific controversy about validation techniques in structural
biology.
This paper provides a comprehensive statistical analysis of the EfN phenomenon
above. We show that the Fourier phases of the EfN estimator (namely, the average of
the aligned noise observations) converge to the Fourier phases of the template signal,
thereby explaining the observed structural similarity. Additionally, we prove that the
convergence rate of Fourier phases is inversely proportional to the number of noise
observations and, in the high-dimensional regime, to the Fourier magnitudes of the
template signal. Moreover, in the high-dimensional regime, the EfN estimator con-
verges to a scaled version of the template signal.
This work not only deepens the
theoretical understanding of the EfN phenomenon but also highlights potential pitfalls
in template matching techniques and emphasizes the need for careful interpretation of
noisy observations across disciplines in engineering, statistics, physics, and biology.
∗Corresponding author: amnonba15@gmail.com
1
arXiv:2407.05277v3  [eess.SP]  10 Mar 2026

## Page 2

Contents
1
Introduction
4
2
Problem Formulation and Notation
6
3
Cryo-EM and Empirical Demonstration
8
4
Main Results
11
4.1
Finite-dimensional signal . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
11
4.2
High-dimensional regime . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
12
5
Extension to other noise statistics
15
5.1
Positive correlation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16
5.2
High-dimensional i.i.d. noise . . . . . . . . . . . . . . . . . . . . . . . . . . .
16
5.3
Circulant Gaussian process . . . . . . . . . . . . . . . . . . . . . . . . . . . .
17
6
Discussion and outlook
19
6.1
Extensions and implications . . . . . . . . . . . . . . . . . . . . . . . . . . .
19
6.2
Future work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21
Appendices
28
Appendix A Preliminaries
28
A.1 Notations
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28
A.2 The convergence of the Einstein from Noise estimator . . . . . . . . . . . . .
29
A.3 Conditioning on the Fourier frequency noise component . . . . . . . . . . . .
30
A.4 Uniqueness of the maximizer . . . . . . . . . . . . . . . . . . . . . . . . . . .
33
A.5 Positive probability of each maximizer event . . . . . . . . . . . . . . . . . .
34
A.6 Auxiliary result for Proposition B.2 . . . . . . . . . . . . . . . . . . . . . . .
36
Appendix B Proof of Theorem 4.1
41
B.1 Convergence of the Fourier phases . . . . . . . . . . . . . . . . . . . . . . . .
42
B.2 Convergence to non-vanishing signal . . . . . . . . . . . . . . . . . . . . . . .
43
B.3 Convergence rate in distribution of the Fourier phases . . . . . . . . . . . . .
44
B.4 Convergence rate in expectation of the Fourier phases . . . . . . . . . . . . .
45
B.5 Proof of Theorem 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
49
Appendix C High-dimensional argmax asymptotics
50
C.1 Proof of Lemma C.2
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
51
C.2 Proof of Proposition C.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
53
Appendix D Proof of Theorem 4.3
56
D.1 Notations and auxiliary results . . . . . . . . . . . . . . . . . . . . . . . . . .
56
D.2 High-dimensional limits . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
58
D.3 Proof of Theorem 4.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
62
2

## Page 3

Appendix E Proof of Proposition 5.1
63
Appendix F Proof of Theorem 5.2: High-dimensional i.i.d. noise
65
F.1
The functional CLT for DFT . . . . . . . . . . . . . . . . . . . . . . . . . . .
66
F.2
Notations
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
66
F.3
Convergence of the real and imaginary parts of the EfN estimator . . . . . .
67
F.4
Proof of Theorem 5.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
69
Appendix G Proof of Proposition 5.4: Circulant Gaussian noise
71
G.1 Preliminaries
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
71
G.2 The convergence of the Einstein from Noise estimator . . . . . . . . . . . . .
72
G.3 Conditioning on the Fourier frequency noise component . . . . . . . . . . . .
73
G.4 Convergence of the Fourier phases . . . . . . . . . . . . . . . . . . . . . . . .
76
G.5 Convergence to non-vanishing signal . . . . . . . . . . . . . . . . . . . . . . .
77
G.6 Proof of Proposition 5.4
. . . . . . . . . . . . . . . . . . . . . . . . . . . . .
78
3

## Page 4

1
Introduction
Model bias is a fundamental pitfall arising across a broad range of statistical problems,
leading to consistent but inaccurate estimations due to systematic errors in the model. This
paper focuses on the Einstein from Noise (EfN) experiment: a prototype example of model
bias that appears in template matching techniques. Consider a scenario where scientists
acquire observational data and genuinely believe their observations contain noisy, shifted
copies of a known template signal. However, in reality, their data consists of pure noise with
no actual signal present.
To estimate the (absent) signal, the scientists align each observation by cross-correlating
it with the template and then average the aligned observations.
Remarkably, empirical
evidence has shown, multiple times, that the reconstructed structure from this process is
structurally similar to the template, even when all the measurements are pure noise [23, 44,
46]. This phenomenon stands in striking contrast to the prediction of the unbiased model,
that averaging pure noise signals would converge towards a signal of zeros, as the number
of noisy observations diverges. Thus, the above EfN estimation procedure is biased towards
the template signal.
While the EfN phenomenon has been analyzed in prior work (see Section 3 for more
details), a comprehensive theoretical understanding of the EfN model remains limited. This
work contributes to filling that gap by rigorously analyzing the relationship between the
reconstructed signal and the underlying template.
The term ’Einstein from Noise’ was
popularized in [44], where the authors illustrated the phenomenon using an image of Einstein
as the template signal.
However, the underlying effect had been observed earlier in the
cryogenic electron microscope literature (see, for instance, [46], and further details in Section
3). In this work, we refer to the average of the aligned pure noise observations as the EfN
estimator. A detailed formulation of the problem is provided in Section 2.
Figure 1 illustrates the EfN process, which consists of two key stages. First, the observa-
tions are aligned with the template signal to achieve optimal alignment. Then, the aligned
observations are averaged. The result is the EfN estimator, which shares a structural resem-
blance to the template image, though it is not an identical reproduction.
Main results.
The central results of this work are as follows. Our first result, stated in
Theorem 4.1, shows that the Fourier phases of the EfN estimator converge to the Fourier
phases of the template signal, as the number of noisy observations (denoted by M) converges
to infinity. However, it is important to note that the EfN estimator’s Fourier magnitudes
do not necessarily converge to those of the template signal. We also show that the Fourier
phases’ mean squared error (MSE) decays to zero with a rate of 1/M. Since the Fourier
phases are responsible for the formation of geometrical image elements, such as contours
and edges [36, 45], this clarifies why the resulting EfN estimator image exhibits a structural
similarity to the template, but not necessarily a full recovery. Our second result, stated in
Theorem 4.3, proves that in the high-dimensional regime, where the dimension of the signal
diverges, the convergence rate of the Fourier phases is inversely proportional to the square
of the Fourier magnitudes of the template signal. In this case, the Fourier magnitudes of the
EfN estimator converge to a scaled version of the template’s Fourier magnitudes.
While Theorems 4.1 and 4.3 are proved under the assumption of white Gaussian noise, we
4

## Page 5

Find location of 
cross-correlation 
maximum (෡R𝑖)
Shift 𝑛𝑖 by 
−෡R𝑖 
Average aligned 
noise images
𝑛𝑖𝑖=0
𝑀−1
𝑛𝑖, ෡R𝑖𝑖=0
𝑀−1
Input: 
Einstein template 
Output: 
Einstein from Noise
Alignment process
Input: 
𝑀 noise images
𝑛𝑖𝑖=0
𝑀−1
ො𝑥= 1
𝑀෍
𝑖=0
𝑀−1
𝑇෡R𝑖
−1𝑛𝑖
𝑇෡R𝑖
−1𝑛𝑖𝑖=0
𝑀−1
𝑥
Figure 1: Einstein from Noise. The EfN estimator consists of three stages: (1) finding the index
of the maximum of the cross-correlation (ˆRi) between the i-th noise signal (ni) and the template
signal (e.g., Einstein’s image); (2) cyclically shifting the noise signal by −ˆRi; (3) averaging the
shifted noise signals. In this paper, we characterize the relationship between the output of this
process—the EfN estimator—and the template signal.
also extend our analysis to more general noise models. In particular, we show that, although
the convergence results in Theorems 4.1 and 4.3 do not necessarily hold under arbitrary noise
statistics, several structural properties of the EfN estimator persist. First, in Proposition 5.1,
we show that the EfN estimator remains positively correlated with the template for arbitrary
noise statistics, even when the Fourier phases do not converge. Since the correlation between
images often implies visual resemblance, this explains why the EfN estimator still exhibits
structural similarity to the template. Second, in Theorem 5.2, we show that in the high-
dimensional limit, if the noise signal is independent and identically distributed (i.i.d) (not
necessarily Gaussian), then the same phase convergence behavior observed in the white
Gaussian case still holds. Finally, in Proposition 5.4, we demonstrate that if the noise signal
is Gaussian with circular symmetry, then the conclusions of Theorem 4.1 remain valid, even
though the noise is not white.
Organization.
The remainder of this paper is organized as follows. Section 2 provides a
detailed formulation of the problem. Section 3 discusses the connection between the EfN
problem and single-particle cryo-electron microscopy (cryo-EM), the primary motivation
for this work, and presents supporting empirical demonstrations.
Our main theoretical
results for white Gaussian noise observations, Theorems 4.1 and 4.3, are stated in Section 4.
Extensions of these results to noise models beyond white Gaussian noise are presented in
Section 5. Finally, we conclude with a discussion and outlook in Section 6.
5

## Page 6

2
Problem Formulation and Notation
This section outlines the probabilistic model behind the EfN experiment and delineates our
main mathematical objectives. Although the EfN phenomenon is described typically for
images, we will formulate and analyze it for one-dimensional signals, bearing in mind that
the extension to two-dimensional images is straightforward (see Section 6 for more details).
Notations.
Throughout the rest of this paper, we use
D−→,
P−→,
a.s.
−−→, and
Lp
−→, to denote the
convergence of sequences of random variables in distribution, in probability, almost surely,
and in Lp norm, respectively. Inner products in the Euclidean space between vectors a and
b are written as either a⊤b or ⟨a, b⟩.
Problem formulation.
Consider a scenario where scientists collect a series of observations
under the belief that each observation is a noisy, randomly shifted version of a known template
signal x ∈Rd (for example, an image of Einstein). Formally, the assumed postulated data
model is given by:
(Postulated model)
yi = Tℓi · x + ni,
(2.1)
where Tℓ: Rd →Rd is the cyclic shift operator defined by [Tℓz]r ≜z(r−ℓ) mod d for all z ∈Rd
and indices 0 ≤r ≤d−1, and ni ∼N(0, σ2Id×d) are i.i.d. Gaussian noise vectors. In reality,
however, there is no underlying signal: the observations consist entirely of white Gaussian
noise. That is, the true data-generating process follows the underlying model:
(Underlying model)
y0, y1, . . . , yM−1
i.i.d.
∼N(0, σ2Id×d),
(2.2)
where M denotes the number of observations. Since the data consists purely of white Gaus-
sian noise, we will explicitly write yi = ni to emphasize this fact.
To estimate the (nonexistent) signal, the scientists align each observation to the template
x using cross-correlation, and then average the aligned observations. Specifically, for each
i = 0, . . . , M −1, they compute the shift that maximizes the inner product with the template:
ˆRi ≜arg max
0≤ℓ<d
⟨ni, Tℓx⟩,
(2.3)
where ni is the i-th noise observation, and ˆRi defines the optimal cyclic shift that aligns the
template signal x with the noise observation ni in terms of cross-correlation.
Then, the EfN estimator is given by the average of the noise observations, but each is
first aligned according to the above maximal shifts, i.e.,
ˆx ≜1
M
M−1
X
i=0
T−ˆRini,
(2.4)
where T−ˆRini represents the noise observation ni aligned by applying the inverse cyclic shift
−ˆRi to best match the template signal. Throughout the text, we refer to ˆx as the EfN
estimator.
The EfN phenomenon states that, at least empirically, ˆx and x appear “close” in some
sense; our goal is to understand this phenomenon mathematically. To that end, we will
6

## Page 7

consider the two asymptotic regimes: the first corresponds to the classical setting where
the number of observations M →∞while the dimension d is fixed (i.e., the number of
observations diverges to infinity and the template vector dimension is fixed); the second
is the high-dimensional regime, where d →∞after M →∞. (i.e., both the number of
observations and the dimension of the template signal diverge).
Fourier space notation.
As will become clear in the next sections, it is convenient to
work in the Fourier domain. Let ϕZ ≜∢Z denote the phase of a complex number Z ∈C,
and recall that the discrete Fourier transform (DFT) of a d-length signal y ∈Rd is given by,
Y[k] ≜F {y} =
1
√
d
d−1
X
ℓ=0
yℓe−j 2π
d kℓ,
(2.5)
where j ≜√−1, and 0 ≤k ≤d −1. Accordingly, we let X, ˆX, and Ni, denote the DFTs
of x, ˆx, and ni, respectively, for 0 ≤i ≤M −1. These DFT sequences can be equivalently
represented in the magnitude-phase domain as follows,
X = {|X[k]| ejϕX[k]}d−1
k=0,
ˆX = {|ˆX[k]|ejϕˆX[k]}d−1
k=0,
Ni = {|Ni[k]| ejϕNi[k]}d−1
k=0,
(2.6)
for 0 ≤i ≤M−1, where |X[k]|,
ˆX[k]
, and |Ni[k]| are the k-th Fourier component magnitudes
of the template signal, the EfN
estimator, and the i-th noise observation, respectively.
Similarly, ϕX[k], ϕˆX[k], and ϕNi[k] represent the corresponding k-th Fourier phases. Note
that the random variables {|Ni[k]|}d/2
k=0 and {ϕNi[k]}d/2
k=0 are two independent sequences of
i.i.d. random variables, such that, |Ni[k]| ∼Rayleigh (σ2) has Rayleigh distribution, and the
phase ϕNi[k] ∼Unif[−π, π) is uniformly distributed over [−π, π).
With the definitions above, we can express the estimation process in the Fourier domain.
Since a shift in real-space corresponds to a linear phase shift in the Fourier space, it follows
that,
ˆX[k] = 1
M
M−1
X
i=0
|Ni[k]| ejϕNi[k]ej 2πk
d ˆRi,
(2.7)
for k = 0, 1, . . . , d −1, where |Ni[k]| and ϕNi[k] are defined in (2.6).
It is important to
note that the expression above will converge to zero without the last term that captures the
dependency in ˆRi—the location of the maximum correlation. This term reflects the funda-
mental properties of the EfN process and its dependency on the template signal, as well as
the connections between the different spectral components. We denote by E|ϕˆX[k] −ϕX[k]|2
the MSE of the Fourier phases of the k-th spectral component.
Assumptions.
Throughout this paper, we assume that the template signal x is normal-
ized, i.e., ∥x∥2
2 = 1, where ∥·∥2 is the Euclidean norm, and further assume that its Fourier
transform in non-vanishing, except possibly at the DC (zero-frequency) component. The
first assumption is used for convenience and does not alter (up to a normalization factor)
our main results in Theorems 4.1 and 4.3. The second assumption is essential for the theo-
retical analysis of the EfN process and is expected to hold in many applications, including
cryo-EM. A similar assumption is frequently taken in related work, e.g., [8, 38, 10]. It is
7

## Page 8

worth noting that since the Fourier transform of x is assumed to be non-vanishing, the max-
imizing shift ˆRi in (2.3) is almost surely unique. In addition, without loss of generality, we
assume that the signal length d is even.
3
Cryo-EM and Empirical Demonstration
Cryo-EM is a powerful tool of modern structural biology, offering advanced methods to vi-
sualize complex biological macromolecules with ever-increasing precision. One of its central
advantages lies in its capability to resolve the structures of proteins that are hard to crystal-
lize in traditional methods, especially in a near-physiological environment, see e.g., [35, 48].
This advantage enables researchers to delve into the dynamic behaviors of proteins and their
complexes, shedding light on fundamental biological processes.
Single-particle cryo-EM uses electron microscopy to reconstruct 3D structures from 2D
tomographic projection images [9]. Typically, the 3D reconstruction involves two main steps:
detecting and extracting single particle images using a particle picking algorithm, [42, 22,
11, 21], and then reconstructing the 3D density map [41, 40]. Most detection algorithms use
template-matching techniques, which can introduce bias if improper templates are chosen,
especially in low signal-to-noise ratio (SNR) conditions, which is the standard scenario in
cryo-EM.
The EfN controversy.
A publication of the 3D structure of an HIV molecule in PNAS
in 2013 [33] initiated a fundamental controversy about validation techniques within the
cryo-EM community, published as four follow-up PNAS publications [23, 54, 52, 32]. The
EfN pitfall played a central role in this discussion. The primary question of the discussion
was whether the collected datasets contained informative biological data or merely pure
noise images.
The core of the debate emphasized the importance of exercising caution
and implementing cross-validation techniques when fitting data to a predefined model. This
precautionary approach aims to mitigate the risk of erroneous fittings, which could ultimately
lead to inaccuracies in 3D density map reconstruction. Model bias is still a fundamental
problem in cryo-EM, as highlighted by an ongoing debate concerning validation tools, see
for example, [51, 44, 24, 16, 17, 25, 27, 50].
Empirical demonstration.
As introduced in the previous section, the EfN phenomenon
depends on several key parameters: (1) the number of observations, denoted by M; (2) the
dimension of the signal, denoted as d (for example, the number of pixels in Einstein’s image);
and (3) the statistical properties of the template signal, and in particular its power spectral
density (PSD). To demonstrate the dependency on these parameters and provide insight into
our main results, Figures 2 and 3 show the convergence of the EfN estimator. Both figures
were generated according to the procedure outlined in Section 2. When referring to Monte
Carlo trials, it means that the EfN estimator procedure, as specified in (2.4), was executed
multiple times (the number of Monte-Carlo trials), each trial with fresh data.
Figure 2 shows the EfN estimator as a function of M. Figure 2(a) shows that as M
increases, the EfN estimator becomes more structurally similar to the template Einstein
image. Indeed, Figure 2(b) shows that as the number of observations M grows, the MSE
between the Fourier phases of the template image and the corresponding Fourier phases of
8

## Page 9

EfN estimator decreases. Figure 2(c) highlights that the convergence rate is proportional to
1/M, with a faster convergence rate for stronger spectral components.
Figure 3 illustrates the impact of the template signal’s PSD on the cross-correlation
between the template signal and the EfN estimator. Notably, a flatter PSD (i.e., a faster
decay of the auto-correlation) leads to a higher correlation between the template and the
estimator signals. These empirical results are proved theoretically in Theorems 4.1 and 4.3.
More applications.
The EfN phenomenon extends to various applications employing tem-
plate matching, whether through a feature-based or direct template-based approach. For
instance, template matching holds significance in computational anatomy, where it aids in
discovering unknown diffeomorphism to align a template image with a target image [15].
Other areas include medical imaging processing [1], manufacturing quality control [3], and
navigation systems for mobile robots [28]. This pitfall may also arise in the feature-based
approach, which relies on extracting image features like shapes, textures, and colors to match
a target image by neural networks and deep-learning classifiers [59, 34, 53, 30].
Previous work.
The EfN phenomenon has been investigated in earlier studies. In partic-
ular, it was shown that the ratios between the expected values of the Fourier coefficients of
the EfN estimator and those of the template are real-valued [60, Chapter 5]. In this work,
we build upon and significantly extend these results. Specifically, we establish the conver-
gence of the EfN estimator to a non-vanishing signal, derive its convergence rate, analyze its
behavior in the high-dimensional regime, and generalize the analysis to encompass a broader
class of noise models beyond white Gaussian noise.
A closely related work is that of Wang et al. [55], who conducted a rigorous statistical
analysis of model bias in a different but complementary setting. They analyze the effects
of selectively averaging only samples that exhibit the highest cross-correlation with a fixed
reference signal (e.g., Einstein’s image). This selection mechanism introduces a bias toward
the reference, and their analysis reveals a phase transition in the resulting reconstruction,
governed by the number of samples, the signal dimension, and the size of the selected subset.
Notably, their results show that a structured image can emerge even when averaging purely
noisy data. A related but statistically distinct selection-based mechanism is studied in [7] in
the context of template matching for particle picking and extraction: candidate (pure-noise)
observations are filtered by thresholding their cross-correlation with a bank of templates,
and downstream averages inherit a strong template imprint.
In contrast, our work investigates the EfN estimator in the absence of any selection
step: all pure-noise observations are first aligned to a fixed template and then averaged.
This difference in mechanism leads to different statistical behavior.
Whereas template-
based selection can yield averages that closely resemble the templates up to a global scale
factor, we show that the non-selective EfN procedure exhibits a more structured form of
alignment-induced bias, manifested most prominently through phase locking: the Fourier
phases of the estimator converge to those of the template, even though the resulting average
is generally not identical to the template. Closely related alignment-induced artifacts and
their connection to Fourier-phase behavior have also been discussed in the multireference
alignment literature; see, e.g., [43].
9

## Page 10

𝑀= 200
𝑀= 500
𝑀= 1500
𝑀= 5000
Template
Template’s PSD
(a)
(b)
100
10−1
10−2
10−3
0
𝑘1
100
-100
𝑘1
𝑘2
MSE
0
100
-100
0
100
-100
0
100
-100
𝑘1
𝑘1
10−1
10−2
10−3
10−4
0
𝑘1
100
-100
0
-100
100
Fourier space phases 𝜙෡X −𝜙X
2
PSD
(c)
10−5
10−4
10−3
10−2
10−1
100
103
104
Number of samples (𝑀)
𝑋1,0
> 𝑋1,1
> 𝑋2,2
> 𝑋3,3
Fourier phases
𝜙෡X −𝜙X
2
Figure 2: The impact of the number of noise observations on the EfN estimator. The
EfN estimator is defined in real space by (2.4) and in Fourier space by (2.7). (a) The structural
similarity between the EfN estimator and the template image increases as a function of the number
of noise observations (M). (b) The mean-square-error (MSE) between the Fourier phases of the
template image X[k1, k2] and the EfN estimator ˆX[k1, k2] for −100 ≤k1, k2 ≤100, where k1, k2 are
the indices of the 2D DFT. The colors in the left panel in (b) represent the power spectral density
(PSD) of the Einstein image, while the colors in the four right panels represent the MSE between
the Fourier phases of the Einstein image and the EfN estimator, for each spectral component,
with a varying number of observations (M = 200, 500, 1500, 5000). An increase in the number of
observations leads to a lower MSE of the Fourier phases between the EfN estimator and the template
signal. A similar trend can be seen with respect to the strength of the spectral components, i.e.,
stronger spectral components lead to lower Fourier phases MSE. (c) The convergence rate of the
MSE between the Fourier phases of the EfN estimator and the Fourier phases of the template signal
as a function of the number of observations across different frequencies. The MSE decays as 1/M.
In addition, stronger spectral components lead to lower MSE. Figures (b) and (c) were generated
through 200 Monte-Carlo trials of the EfN process defined in (2.4).
10

## Page 11

0 𝑘1
𝑘20
0.05
0
X 𝑘1, 𝑘2
෡X 𝑘1, 𝑘2
𝑘1
𝑘2
0
0
0.05
0
0.12
0.12
X 𝑘1, 𝑘2
෡X 𝑘1, 𝑘2
𝑘1
𝑘2
𝑘2
0
0
0
0
Fourier space magnitudes of template image and EfN estimator
0
0
0
0
0
0
𝑘1
𝑘2
𝑘2
0
0
0.3
0.3
X 𝑘1, 𝑘2
෡X 𝑘1, 𝑘2
(a)
(b)
Template: Flatter power spectral density, faster auto-correlation decay
EfN estimator: Higher cross-correlation with the template
CC = 0.87
CC = 0.95
CC = 0.99
Figure 3: The influence of the power-spectral-density (PSD) of the template signal
on the correlation between the template and the EfN estimator. (a) Three images of
the letter A are shown, with an increasing zero-padding ratio. As the zero-padding ratio increases,
the PSD flattens, and the cross-correlation (CC) between the template and the EfN estimator
increases. This higher cross-correlation is evident in both the image background and the colors of
the letter A. (b) Flatter PSDs lead to EfN estimators whose Fourier magnitudes are closer to those
of the template image. The EfN estimators in these experiments were generated using M = 105
observations.
4
Main Results
We begin by analyzing the regime where M →∞and the dimension of the signal d is fixed.
In this setting, we show that the Fourier phases of the EfN estimator converge almost surely
to those of the underlying template signal, and we characterize the convergence rate. We
also analyze the behavior of the Fourier magnitudes. Then, we turn to the high-dimensional
regime, where d →∞. Under additional assumptions, we derive refined asymptotic guaran-
tees for both the phases and magnitudes. Throughout, we assume that the template signal
x ∈Rd has a unit norm and that its spectrum is non-vanishing, i.e., X[k] ̸= 0 for every
0 < k ≤d −1, as discussed in the previous section.
4.1
Finite-dimensional signal
We begin with the case where the template signal has a fixed dimension d, as captured in
the following result, whose proof is provided in Appendix B.5.
Theorem 4.1 (Fourier phases convergence for finite-dimensional signal). Fix d ≥2 and
assume that X[k] ̸= 0, for all 0 < k ≤d −1.
11

## Page 12

1. For any 0 ≤k ≤d −1, we have,
ϕˆX[k]
a.s.
−−→ϕX[k],
(4.1)
as M →∞. Furthermore,
lim
M→∞
E|ϕˆX[k] −ϕX[k]|2
1/M
= Ck,
(4.2)
for a finite constant Ck < ∞.
2. For any 0 ≤k ≤d −1, we have,
|ˆX[k]|
a.s.
−−→E

|N[k]| cos
2πk
d
ˆR1 + ϕN[k] −ϕX[k]

> 0,
(4.3)
as M →∞, where ˆR1 is defined in (2.3).
Theorem 4.1 captures two central properties.
The first addresses the convergence of
the EfN estimator’s phases to those of the template signal. In addition, the corresponding
convergence rate in MSE is proportional to 1/M. The second result captures the convergence
of the EfN estimator’s magnitudes to the term given in the right-hand-side (r.h.s.) of (4.3),
which is strictly greater than zero. Thus, the EfN estimator converges to a non-vanishing
signal. Interestingly, this term is not necessarily proportional to the magnitudes |X[k]| of the
template signal and, thus, the EfN estimator reproduces (asymptotically) only the phases of
the template signal but not the magnitudes.
A central component of the proof of Theorem 4.1 is the circulant structure inherent in
the alignment of the noise, which arises from the cyclic shift operations. This symmetry
implies that the covariance matrix of the noise-aligned sum is circulant, corresponding to a
cyclo-stationary Gaussian process. In particular, we apply the central limit theorem (CLT)
and the strong law of large numbers (SLLN) for this setting, which yields
ϕˆX[k] −ϕX[k]
D−→arctan(Qk),
as M →∞, where Qk is a zero-mean Gaussian random variable with variance σ2
Q[k] = Ck/M,
and the constant Ck admits a closed-form expression. By leveraging properties of cyclo-
stationary Gaussian processes, which is justified by the circulant structure of the problem,
we establish that Ck < ∞for all 0 ≤k ≤d −1. This directly leads to the results stated in
(4.1)–(4.2).
4.2
High-dimensional regime
We now turn to the high-dimensional setting where d →∞, taken after the limit M →∞. In
this regime, we impose additional technical conditions on the template signal, formalized in
Assumption 4.2. Intuitively, these conditions reflect the empirical phenomenon illustrated in
Figure 3, where a flatter PSD, which corresponds to a more rapidly decaying autocorrelation
function, results in an improved alignment between the template and the estimator.
12

## Page 13

More precisely, Assumption 4.2 below requires control over the decay of both the autocor-
relation function and the spectral magnitudes as functions of d. Specifically, for a length-d
signal x ∈Rd, we define the (circular) autocorrelation in the time domain by
RXX[τ] ≜
1
√
d
d−1
X
n=0
x[n] x[n + τ (mod d)],
(4.4)
for τ ∈{0, 1, . . . , d −1}. By the discrete Wiener-Khinchin theorem, this is equivalent to
taking the inverse discrete Fourier transform of the PSD, namely,
RXX[τ] = F−1
|X|2 	
[τ] =
1
√
d
d−1
X
k=0
|X[k]|2 ej 2π
d kτ,
(4.5)
where X = F{x} is the DFT of x. Assumption 4.2 requires that the autocorrelation RXX[τ]
decay faster than 1/ log d, and that the maximum magnitude among nonzero Fourier compo-
nents maxk̸=0 |X[k]| decay faster than 1/√log d. In addition, we assume the DC component
is vanishing, i.e., |X[0]| = 0, to avoid degeneracies in alignment.
Assumption 4.2 (High-dimensional regularity of the template). Consider a sequence of
template signals {x(d)}d∈N with x(d) ∈Rd, and let X(d) = F{x(d)} denote their DFT. Let
R(d)
XX denote the autocorrelation of x(d), as defined in (4.4). When taking the limit d →∞,
we assume the signals are normalized, namely ∥x(d)∥2 = 1 for all d ∈N. We say that the
template sequence {x(d)}d∈N satisfies Assumption 4.2 if the following hold:
1. Autocorrelation decay. The autocorrelation away from the zero lag satisfies
lim
d→∞

max
1≤τ≤d−1
R(d)
XX[τ]


· log d = 0.
(4.6)
2. Spectral magnitude decay. The non-DC Fourier magnitudes satisfy
lim
d→∞

max
1≤k≤d−1
X(d)[k]


·
p
log d = 0.
(4.7)
3. Vanishing DC component. The signal’s DC component is zero, i.e.,
X(d)[0]
 = 0.
Although the conditions in Assumption 4.2 may seem technical, they are essential for
establishing Theorem 4.3, which relies on classical limit theorems for the maxima of station-
ary Gaussian processes, most notably, convergence to the Gumbel distribution [29, 12, 2, 4].
Each part of the assumption plays a specific role: Part (1) ensures that the noise process lacks
long-range dependencies, which corresponds to a sufficiently flat PSD; Part (2) guarantees
that no individual Fourier component dominates the behavior of the EfN estimator. The final
condition, requiring the DC component to vanish (i.e., |X(d)[0]| = 0), is not strictly necessary
from an empirical standpoint but is introduced to streamline the theoretical analysis.
Theorem 4.3 (Fourier phases convergence for high-dimensional signal). Assume that X(d)[k] ̸=
0, for all 0 < k ≤d −1, and that x satisfies Assumption 4.2. Then,
13

## Page 14

1. For any 0 ≤k ≤d −1, we have,
lim
d→∞lim
M→∞
E|ϕˆX(d)[k] −ϕX(d)[k]|2
1/(M log d)
1
1/(4 |X(d)[k]|2)
= 1.
(4.8)
2. For any 0 ≤k ≤d −1, we have,
1
σ√2 log d
|ˆX(d)[k]|
|X(d)[k]|
a.s.
−−→1,
(4.9)
as M, d →∞.
The proof of Theorem 4.3 is presented in Appendix D.3. Based on Theorem 4.3, as
M, d →∞, the convergence rate of the Fourier phases of the EfN estimator is inversely
proportional to the squared Fourier magnitude. In addition, unlike the fixed-d result in
Theorem 4.1, the high-dimensional regime reveals an explicit dependence on d in the phase-
error constant. Specifically, while Theorem 4.1 states that for each fixed dimension d, (4.2)
holds with a finite constant Ck (which depends on d and on the template), Theorem 4.3
makes this dependence explicit when d →∞: comparing (4.8) with (4.2) shows that in the
high-dimensional limit the constant scales as
Ck =
1
4 |X(d)[k]|2 log d.
(4.10)
Intuitively, the log d factor arises from the alignment step: ˆRi is chosen by maximizing a
stationary Gaussian correlation process over d shifts, and the maximizer is governed by
extreme-value statistics. In particular, the maximum of such a process grows on the scale
√2 log d; see, e.g., [29, 12, 2, 4], and the proof in Appendix D.
Moreover, unlike Theorem 4.1 for a fixed d, Theorem 4.3 also shows that the Fourier
magnitudes of the EfN estimator satisfy (4.9), namely they recover the template magni-
tudes up to the known normalization factor σ√2 log d.
Therefore, when d →∞under
Assumption 4.2, the normalized estimate ˆx(d) recovers the template signal, which in turn
implies that the normalized cross-correlation between the template and the EfN estimator
approaches unity.
Empirically, we observe that Theorem 4.3 provides accurate predictions of the conver-
gence behavior when Assumption 4.2 holds. As illustrated in Figure 4, the convergence rate
is strongly influenced by the PSD of the template signal. In particular, Figure 4(b) shows
that increasing the signal length and a flatter PSD lead to a stronger correlation between
the EfN estimator and the true template. Furthermore, Figure 4(c) demonstrates that the
convergence of the Fourier phases of the EfN estimator aligns closely with the theoretical
predictions as the PSD becomes flatter. When the template violates Assumption 4.2 (e.g., if
its autocorrelation decays too slowly), the predicted convergence rates become less accurate,
highlighting the importance of the assumption for the theorem’s formal guarantees. However,
even when the spectral decay is moderate, and the assumption is not strictly met, we find
that the analytical convergence rates still rather closely match empirical observations (Fig-
ure 4(c)). Notably, the key phenomenon that the convergence rate of the Fourier phases is
inversely related to the magnitude of the corresponding spectral components remains robust
beyond the regime where the theorem formally applies.
14

## Page 15

Monte-Carlo Simulation
Asymptotic expression
Template’s PSD
10−2
0
2000
4000
-2000
Frequency bin
-4000
(a)
(c)
10−6
10−5
10−4
10−3
10−1
Signal-1
Signal-2
Signal-3
0
2000
4000
-2000
-4000
Signal-3
Frequency
10−3
10−2
10−1
100
(b)
0
2000
4000
-2000
-4000
Signal-2
Frequency
10−3
10−2
10−1
100
0
2000
4000
-2000
-4000
Signal-1
Frequency
10−3
10−2
10−1
100
𝔼𝜙෡X −𝜙X 2
Signal length 𝑑
102
103
104
Pearson Cross- Correaltion
PCC 𝑥(𝑑), ො𝑥(𝑑)
Signal-2
Signal-3
0.82
0.84
0.86
0.88
0.9
0.92
0.94
0.96
0.98
1
Fourier phases convergence
Pearson Cross-Correlation
Figure 4: Comparison between analytic expression and Monte-Carlo simulations for
high-dimensional signals, d, and for signals with varying power spectral densities.
The analytic predictions for Fourier-phase convergence and Fourier-magnitude scaling are given by
(4.8) and (4.9), respectively. (a) Template PSDs for three template families at a representative
dimension d = 8192.
For each dimension d, the template x(d) ∈Rd is generated directly at
length d as an exponentially decaying signal, x(d)
ℓ[m] ∝exp(−m/αℓ), m = 0, 1, . . . , d −1, with
decay parameters αℓ∈{0.02, 2, 10} (Signals 1-3, respectively), followed by mean removal and
normalization. (b) Monte-Carlo estimates of the Pearson cross-correlation PCC(x(d)
ℓ, ˆx(d)
ℓ) between
the template x(d)
ℓ
and the EfN estimate ˆx(d)
ℓ
as a function of the signal length d (with fixed sample
size M = 104). As d increases, the correlation increases, particularly for templates with slower-
decaying PSDs. (c) Per-frequency phase mean-squared error E|ϕˆX(d)
ℓ[k] −ϕX(d)
ℓ[k]|2 at d = 8192:
Monte-Carlo estimates (blue) are compared with the asymptotic expression (red), i.e., the large-
(M, d) closed-form approximation predicted by (4.8). All Monte-Carlo curves are averaged over
2000 independent trials.
5
Extension to other noise statistics
So far, we have analyzed the setting in which the noise is white Gaussian. In this section, we
extend the analysis to a broader class of noise distributions. Specifically, we now assume that
the observations y0, y1, . . . , yM−1 ∈Rd are i.i.d. samples drawn from an arbitrary distribution
15

## Page 16

with zero mean and a fixed covariance matrix, namely,
E[y1] = 0,
E[y1y⊤
1 ] = Σ,
(5.1)
where Σ ≻0 is a positive-definite matrix with bounded operator norm, i.e., ∥Σ∥< ∞.
Notably, the entries of each sample yi are not required to be independent or identically
distributed.
5.1
Positive correlation
In general, the Fourier phase convergence property established under the white Gaussian
assumption does not hold for arbitrary noise distributions, as demonstrated empirically in
Figures 5 and 6. Nonetheless, we establish a positive correlation result between the EfN
estimator and the underlying template signal.
Proposition 5.1 (Positive correlation). Let d ≥2, and suppose the observations {yi}M−1
i=0
are drawn i.i.d. according to the model in (5.1). Let x ∈Rd denote the template signal, and
assume its discrete Fourier transform X satisfies X[k] ̸= 0 for all 1 ≤k ≤d −1. Let ˆx be
the EfN estimator computed from the observations {yi}. Then, as M →∞, the following
inequality holds almost surely,
⟨ˆx, x⟩≥
max
0≤r1,r2<d−1
1
2 E [|⟨y1, Tr1x −Tr2x⟩|] > 0.
(5.2)
The proof of Proposition 5.1 is provided in Appendix E. This result implies that the EfN
estimator is positively correlated with the true template signal. Although this is a weaker
guarantee than the Fourier phase convergence obtained under Gaussian white noise, it still
ensures that the estimator retains meaningful structural information from the template.
5.2
High-dimensional i.i.d. noise
Our next result demonstrates that the Fourier phase convergence established in Theorem 4.1
for Gaussian white noise extends to a broader class of noise models in the high-dimensional
regime. To this end, we impose an additional assumption that the entries of each observation
vector yi ∈Rd are i.i.d. Namely, the covariance matrix Σ is diagonal.
Theorem 5.2 (High-dimensional i.i.d. noise). Let {yi}M−1
i=0
be i.i.d. observations drawn
according to the model in (5.1), and assume further that the entries of each yi ∈Rd are
i.i.d., with finite variance, and satisfy E[(yi[ℓ])4] < ∞, for all ℓ∈{0, 1, . . . , d −1}. Let ˆX
denote the discrete Fourier transform of the EfN estimator under this noise model. Assume
that the Fourier coefficients of the template x are non-vanishing, i.e., X[k] ̸= 0 for all k ∈N+.
Then, for any fixed k ∈N+, we have,
ϕˆX[k] −ϕX[k]
a.s.
−−→0,
(5.3)
as M, d →∞. Moreover,
lim
d→∞lim
M→∞
E [|ϕˆX[k] −ϕX[k]|2]
1/M
= Ck,
(5.4)
16

## Page 17

for some finite constant Ck < ∞. Finally, if x satisfies Assumption 4.2, then,
lim
d→∞lim
M→∞
E [|ϕˆX[k] −ϕX[k]|2]
1/(M log d)
·
1
1/(4|X[k]|2) = 1.
(5.5)
The proof of Theorem 5.2 is given in Appendix F. In essence, this result extends the
Fourier phase convergence of Theorem 4.1 to a broader class of noise distributions in the
high-dimensional setting. The main idea of the proof is to apply the functional central limit
theorem to the DFT coefficients [37, 14, 13]. As d →∞, the Fourier components of the
noise converge in distribution to those of a circulant Gaussian random process, owing to the
i.i.d. structure of the entries in yi. This asymptotic Gaussianity enables us to apply the
same analytical framework developed for the white noise case to establish convergence of the
Fourier phases.
Empirical demonstration.
Figure 5 provides empirical validation of Theorem 5.2 in
settings where the noise distribution is non-Gaussian. In particular, we consider yi ∈Rd
with i.i.d. entries drawn from either the uniform or Poisson distribution. As the figure
shows, when d is relatively small, the Fourier phases fail to converge and instead plateau.
However, as the dimension increases, phase convergence emerges at the predicted 1/M rate,
aligning with our theoretical results.
5.3
Circulant Gaussian process
In this section, we consider the setting in which the noise exhibits correlations between
entries. As previously noted, Fourier phase convergence does not generally hold under ar-
bitrary noise models. However, we show that convergence is maintained when the noise
follows a circulant Gaussian distribution, a structured class of Gaussian noise characterized
by rotational symmetry.
Definition 5.3 (Symmetric circulant matrix). A matrix Σ ∈Rd×d is called circulant if
each row is a right cyclic shift of the previous one.
That is, there exists a vector c =
(c0, c1, . . . , cd−1) ∈Rd such that
Σ = circ(c) =


c0
c1
c2
. . .
cd−1
cd−1
c0
c1
. . .
cd−2
...
...
...
...
...
c1
c2
c3
. . .
c0

.
(5.6)
The matrix is said to be symmetric circulant if cj = cd−j for all j = 1, . . . , d −1.
Proposition 5.4 (Fourier phase convergence under circulant Gaussian noise). Let d ≥2 be
fixed, and suppose the observations {yi}M−1
i=0
are i.i.d samples drawn from the multivariate
normal distribution N(0, Σ), where Σ is a symmetric circulant matrix as defined in Defini-
tion 5.3. Assume further that the eigenvalues of Σ are strictly positive, and that the template
signal x ∈Rd satisfies X[k] ̸= 0 for all 1 ≤k ≤d −1. Let ˆx denote the EfN estimator under
this noise model. Then, for each 0 ≤k ≤d −1:
ϕˆX[k]
a.s.
−−→ϕX[k],
(5.7)
17

## Page 18

103
104
10−5
10−4
10−3
10−2
10−1
Plateau
Poisson noise
10−5
10−4
10−3
10−2
10−1
103
104
103
104
Gaussian white noise
𝜙෡X 𝑘−𝜙X 𝑘
2
𝜙෡X 𝑘−𝜙X 𝑘
2
𝑑= 8
Number of observations (𝑀)
Number of observations (𝑀)
𝑑= 1024
103
104
Number of observations (𝑀)
Plateau
Uniform noise
103
104
Number of observations (𝑀)
Number of observations (𝑀)
103
104
Number of observations (𝑀)
X[1]
X[2]
X[3]
X[1]
X[2]
X[3]
X[1] > X[2] > X[3]
X[1] > X[2] > X[3]
𝑑= 32
103
104
Number of observations (𝑀)
10−5
10−4
10−3
10−2
10−1
𝜙෡X 𝑘−𝜙X 𝑘
2
Plateau
X[1] > X[2] > X[3]
103
104
Number of observations (𝑀)
Plateau
103
104
Number of observations (𝑀)
X[1]
X[2]
X[3]
Figure 5: The impact of noise statistics and signal dimension (d) on Fourier phase
convergence. Each panel displays the mean squared error (MSE) between the Fourier phases of
the true template and those estimated by EfN, shown for three representative Fourier components.
The dashed line represents the theoretical 1/M convergence rate. Columns correspond to different
noise distributions: white Gaussian noise, i.i.d. noise drawn from a uniform distribution over the
interval [0, 1], and i.i.d. Poisson noise with parameter λ = 10. Rows correspond to increasing
signal dimensions: d = 8, 32, and 1024. For white Gaussian noise, the Fourier phases converge at
the expected 1/M rate across all signal dimensions, in agreement with Theorem 4.1. In contrast,
under uniform and Poisson noise, the MSE plateaus at low dimensions. However, increasing the
signal dimension restores convergence, even under non-Gaussian noise, consistent with the high-
dimensional regime described in Theorem 5.2.
Notably, for d = 1024, all three noise models
produce similar MSE values across the selected Fourier components, suggesting that their phase
noise statistics become nearly indistinguishable.
Each data point represents an average of 300
Monte Carlo trials.
18

## Page 19

as M →∞. Moreover,
lim
M→∞
E [|ϕˆX[k] −ϕX[k]|2]
1/M
= Ck,
(5.8)
for some finite constant Ck < ∞.
The proof of Proposition 5.4 is given in Appendix G. In essence, this result serves as a
generalization of Theorem 4.1, which considered the case of white Gaussian noise, to the
broader setting of symmetric circulant Gaussian noise. Notably, white noise with covariance
σ2Id×d is a special case of circulant noise, making this extension a natural generalization.
The critical insight here is that circulant covariance matrices remain diagonalizable in the
Fourier basis, which preserves the independence of the DFT coefficients and enables phase
convergence to proceed as in the white Gaussian case.
Empirical demonstration.
Figure 6 presents an empirical comparison of the MSE of the
Fourier phase estimates, as a function of the number of observations M, under three distinct
noise models: (1) white Gaussian noise with covariance Σ = σ2Id×d; (2) Gaussian noise
with a symmetric circulant covariance matrix, as defined in Definition 5.3; and (3) Gaussian
noise with a Toeplitz (but non-circulant) covariance matrix. As shown in the figure, both
the i.i.d. and circulant models exhibit the expected 1/M decay in the phase MSE curve,
though the constants Ck differ, reflecting their distinct covariance structures. In contrast,
under Toeplitz noise, the phase estimates do not converge: the MSE plateaus, and no 1/M
scaling is observed. These results empirically confirm that the convergence of Fourier phases
is tightly linked to the circulant structure of the noise covariance.
6
Discussion and outlook
In this work, we have shown that the Fourier phases of the EfN estimator converge to those
of the template signal for an asymptotic number of observations. Since Fourier phases are
crucial for perceiving image structure, the reconstructed image appears structurally similar
to the template signal, even in cases where the estimator’s spectral magnitudes differ from
those of the template [36, 45].
We have also shown that the Fourier phases of spectral
components with stronger magnitudes converge faster, leading to faster structural similarity
in the overall image perception. In addition, we have extended our analysis beyond white
Gaussian noise, examining other noise models.
We have shown that the EfN estimator
remains positively correlated with the template for arbitrary noise settings, and we have
analyzed the Fourier phases convergence properties for high-dimensional i.i.d. noise (which
is not necessarily Gaussian) and circulant Gaussian noise.
6.1
Extensions and implications
We anticipate that the findings of this paper will be beneficial in various fields. For example,
the paper sheds light on a fundamental pitfall in template matching techniques, which may
lead engineers and statisticians to misleading results. In addition, physicists and biologists
working with data sets of low SNRs will benefit from understanding limitations and potential
19

## Page 20

103
104
10−5
10−4
10−3
10−2
10−1
General Σ 
10−5
10−4
10−3
10−2
10−1
103
104
103
104
Gaussian white noise
𝜙෡X 𝑘−𝜙X 𝑘
2
𝜙෡X 𝑘−𝜙X 𝑘
2
𝑑= 8
Number of observations (𝑀)
Number of observations (𝑀)
𝑑= 1024
103
104
Number of observations (𝑀)
Circulant Σ
103
104
Number of observations (𝑀)
Number of observations (𝑀)
103
104
Number of observations (𝑀)
X[1]
X[2]
X[3]
X[1]
X[2]
X[3]
X[1] > X[2] > X[3]
X[1] > X[2] > X[3]
𝑑= 32
103
104
Number of observations (𝑀)
10−5
10−4
10−3
10−2
10−1
𝜙෡X 𝑘−𝜙X 𝑘
2
X[1] > X[2] > X[3]
103
104
Number of observations (𝑀)
103
104
Number of observations (𝑀)
X[1]
X[2]
X[3]
Plateau
Plateau
Figure 6: The impact of the noise covariance structure and signal dimension (d) on
Fourier phase convergence. Each plot shows the mean squared error (MSE) between the Fourier
phases of the ground-truth template and those estimated by EfN, evaluated across three spectral
components. The dashed line indicates the theoretical 1/M convergence rate. Columns correspond
to three types of noise: (1) white Gaussian noise with covariance Σ = σ2Id×d, (2) symmetric
circulant covariance, and (3) a Toeplitz covariance matrix that is not circulant. Rows represent
increasing signal dimensions: d = 8, 32, and 1024.
Under white Gaussian noise, the Fourier
phases converge at the expected 1/M rate, independent of the signal dimension (Theorem 4.1). A
similar trend is observed when the noise has a circulant covariance structure: the same 1/M scaling
holds, although the MSE is different compared to the white noise case. In contrast, for a Toeplitz
covariance matrix that is not circulant, the MSE plateaus at small signal dimensions, indicating a
failure of convergence. However, when the signal dimension increases to d = 1024, convergence at
the 1/M rate is restored even under this more structured noise model. Each data point represents
an average of 300 Monte Carlo trials.
20

## Page 21

biases introduced by template matching techniques. More generally, this work provides a
cautionary framework for the broader scientific community, highlighting the importance of
exercising care when interpreting noisy observations.
Extension to higher dimensions.
While this paper focuses on one-dimensional signals,
the analysis can be readily extended to higher dimensions. This extension involves replacing
the one-dimensional DFT with its N-dimensional counterpart. The symmetry properties
established in Theorem 4.1, including the results in Propositions B.3 and B.2, remain valid.
For the high-dimensional case of Theorem 4.3, the conditions on the PSD adjust for the
N-dimensional case. Specifically, the auto-correlation decay rate of the multidimensional
array should be faster than 1/log d in each dimension.
Implications to cryo-EM.
The findings of Theorem 4.3 have practical implications for
cryo-EM. Typically, protein spectra exhibit rapid decay at low frequencies (known as the
Guinier plot) and remain relatively constant at high frequencies, a behavior characterized
by Wilson in [58] and known as Wilson statistics. Wilson statistics is used to sharpen 3-D
structures [47]. To mitigate the risk of model bias, we suggest using templates with reduced
high frequencies, recommending filtered, smooth templates. This insight may also relate to
or support the common practice of initializing the expectation-maximization (EM) algorithm
for 3-D refinement with a smooth 3-D volume. Each iteration of the EM algorithm effectively
applies a version of template matching multiple times, although projection images typically
contain actual signal rather than pure noise, as in the EfN case.
The key message for the cryo-EM community is that, regardless of the specific setting,
one should not rely solely on the raw alignment average when working with low-SNR data.
Instead, robust validation practices, such as cross-validation, independent reconstructions,
and other consistency checks, are essential to guard against artifact-driven effects like the
EfN phenomenon. In this context, we mention a recent work suggesting that processing
data in smaller mini-batches can help reduce the risk of EfN, offering a practical approach
to mitigating model bias in such settings [6].
Noise statistics in cryo-EM.
The results in Section 5 are particularly relevant to the
noise characteristics commonly encountered in cryo-EM. While cryo-EM noise is often mod-
eled as Poisson in nature, the standard practical assumption is that it follows a Gaussian
distribution with a decaying power spectrum. These properties align well with the broader
class of noise models considered in our analysis. Consequently, the conclusions of Theo-
rem 5.2 can be extended to the cryo-EM setting, and we expect similar asymptotic phase
convergence behavior to hold.
6.2
Future work
Here, we list open questions and directions for future work.
Extension to non-cyclic group actions.
A natural direction for future work is to extend
the EfN analysis beyond the simplified setting of cyclic translations, as defined in (2.1), to
more general group actions, particularly those arising in practical applications such as cryo-
EM. In this context, the relevant transformations are elements of the rotation group SO(3),
21

## Page 22

and the postulated observations are two-dimensional projections of a three-dimensional struc-
ture rather than simple translations of a one-dimensional signal. Recent empirical evidence
suggests that template-induced bias can persist in such non-abelian settings as well, including
under SO(3) group actions [57].
However, extending the theoretical analysis to non-abelian groups presents more substan-
tial challenges. In particular, the property of circular Gaussian statistics, which underpins
the EfN analysis for cyclic groups, does not naturally extend to the non-abelian setting.
Preliminary simulations for the non-abelian dihedral group (not shown here) indicate that
the convergence of the EfN estimator’s Fourier phases observed in the abelian setting does
not directly carry over.
We nonetheless expect that alignment over a richer group may
still induce systematic structural biases, potentially governed by the group’s representation-
theoretic decomposition. This broader perspective is consistent with recent observations of
confirmation bias effects even in unstructured latent-variable models: for Gaussian mixture
models with pure-noise data, a single iteration of population k-means or EM initialized at
hypothesized centroids produces estimates that remain positively correlated with the ini-
tialization [5], suggesting that algorithmic bias can persist beyond group-alignment settings,
albeit often in weaker forms than the phase-locking phenomenon characterized here.
Hard assignment algorithms and the EM algorithm.
One promising avenue for fu-
ture research involves examining hard-assignment algorithms. These algorithms iteratively
refine estimates of an underlying signal from noisy observations, where the signal is obscured
by high noise (unlike the pure noise scenario in EfN). The process begins by aligning ob-
servations with a template signal in the initial iteration and averaging them to improve the
template for subsequent iterations. A central objective is to understand and characterize the
model bias introduced throughout this iterative process, specifically, how the final output
depends on the initial template. Notably, the results presented in this work can be inter-
preted as describing a single iteration of a hard-assignment algorithm in the limit as the
SNR approaches zero.
Another important direction is investigating the EM algorithm, a cornerstone of cryo-EM
algorithms [41, 40]. EM maximizes the likelihood function of models incorporating nuisance
parameters [19], a topic of significant recent interest [18, 56]. Unlike hard-assignment al-
gorithms, EM operates iteratively as a soft assignment algorithm, assigning probabilities to
various possibilities and computing a weighted average rather than selecting a single optimal
alignment per observation. Further exploration of EM could provide deeper insights into
iterative methodologies in cryo-EM and their associated model biases.
Extension to the non-i.i.d. case.
While Theorem 5.2 assumes that the noise entries
within each observation vector yi are independent and identically distributed, an important
direction for future research is to extend these results to more general noise settings. Specifi-
cally, the analysis could be broadened to cover cases where the noise entries are independent
but not identically distributed, provided that their variances remain uniformly bounded and
a Lindeberg-type condition is fulfilled [20]. Moreover, the framework may apply to noise
that exhibits certain weak dependence structures, such as mixing conditions, allowing the
use of functional central limit theorems and ensuring asymptotic Gaussianity of the Fourier
components [37, 14, 13].
22

## Page 23

Asymptotic regimes.
In this work, we analyzed two asymptotic regimes: (1) M →∞
with fixed d (Theorem 4.1), and (2) M →∞followed by d →∞(Theorem 4.3). These
regimes capture distinct theoretical and practical scenarios. Our approach relies on classical
probabilistic tools in the first limit (M →∞), such as SLLN and CLT, and results from the
theory of Gaussian extremes (e.g., convergence to the Gumbel distribution) in the second
(d →∞).
Other challenging asymptotic regimes merit further investigation.
In particular, it is
of interest to understand the behavior in the joint high-dimensional regime where both
M, d →∞with a fixed ratio, i.e.,
d
M →c ∈(0, ∞). This regime, common in modern high-
dimensional statistics, differs from the sequential limits we analyze. More broadly, other
asymptotic behaviors of (M, d) are possible. When both M = Mn and d = dn vary according
to general sequences, a variety of additional regimes may arise, each potentially requiring
different analytical techniques. Typically, in such settings, classical limit theorems may no
longer apply directly, and new challenges arise, such as subtle phase transitions in statistical
behavior and the breakdown of averaging effects when d and M grow at comparable rates.
Addressing these phenomena typically requires more advanced tools from high-dimensional
probability. We view the analysis of further asymptotic settings as a valuable direction for
future research.
Statistical inference.
While the present work establishes the asymptotic consistency of
the EfN estimator’s Fourier phases, an important direction for future research is to investi-
gate their behavior in the finite-sample regime. In particular, developing tools for statistical
inference, such as confidence intervals or non-asymptotic error bounds, would enhance the
practical utility of the analysis. Addressing these questions may require the use of sharper
probabilistic techniques beyond classical limit theorems, such as Berry–Esseen-type results,
concentration inequalities, or non-asymptotic deviation bounds tailored to the specific struc-
ture of the problem.
Acknowledgment
T.B. is supported in part by BSF under Grant 2020159, in part by NSF-BSF under Grant
2024791, and in part by ISF under Grant 1924/21. W.H. is supported by ISF under Grant
1734/21.
References
[1] Ashley Aberneithy. Automatic detection of calcified nodules of patients with tubercu-
lous. University College, London, 2007.
[2] Robert J Adler and Jonathan E Taylor. Random fields and geometry. Springer Science
& Business Media, 2009.
[3] MS Aksoy, Orhan Torkul, and Ismail Hakki Cedimoglu. An industrial visual inspection
system that uses inductive learning. Journal of Intelligent Manufacturing, 15:569–574,
2004.
23

## Page 24

[4] Jean-Marc Aza¨ıs and Mario Wschebor. Level sets and extrema of random processes and
fields. John Wiley & Sons, 2009.
[5] Amnon Balanov, Tamir Bendory, and Wasim Huleihel. Confirmation bias in Gaussian
Mixture Models. IEEE Transactions on Information Theory, 71(11):8871–8898, 2025.
[6] Amnon Balanov, Wasim Huleihel, and Tamir Bendory. Expectation-maximization for
low-SNR multi-reference alignment. arXiv preprint arXiv:2505.21435, 2026.
[7] Amnon Balanov, Alon Zabatani, and Tamir Bendory. Structure from noise: Confir-
mation bias in particle picking in structural biology. arXiv preprint arXiv:2507.03951,
2025.
[8] Afonso S Bandeira, Ben Blum-Smith, Joe Kileel, Jonathan Niles-Weed, Amelia Perry,
and Alexander S Wein. Estimation under group actions: recovering orbits from invari-
ants. Applied and Computational Harmonic Analysis, 66:236–319, 2023.
[9] Tamir Bendory, Alberto Bartesaghi, and Amit Singer.
Single-particle cryo-electron
microscopy: Mathematical theory, computational challenges, and opportunities. IEEE
signal processing magazine, 37(2):58–76, 2020.
[10] Tamir Bendory, Nicolas Boumal, Chao Ma, Zhizhen Zhao, and Amit Singer. Bispectrum
inversion with application to multireference alignment. IEEE Transactions on signal
processing, 66(4):1037–1050, 2017.
[11] Tristan Bepler, Andrew Morin, Micah Rapp, Julia Brasch, Lawrence Shapiro, Alex J
Noble, and Bonnie Berger. Positive-unlabeled convolutional neural networks for particle
picking in cryo-electron micrographs. Nature methods, 16(11):1153–1160, 2019.
[12] Simeon M Berman. Limit theorems for the maximum term in stationary sequences. The
Annals of Mathematical Statistics, pages 502–516, 1964.
[13] David R Brillinger. Time series: data analysis and theory. SIAM, 2001.
[14] Cl´ement Cerovecki and Siegfried H¨ormann. On the CLT for discrete fourier transforms
of functional time series. Journal of multivariate analysis, 154:282–295, 2017.
[15] Gary E Christensen, Richard D Rabbitt, and Michael I Miller. Deformable templates
using large deformation kinematics. IEEE transactions on image processing, 5(10):1435–
1447, 1996.
[16] Jon Cohen. Is high-tech view of HIV too good to be true?, 2013.
[17] Pilar Cossio. Need for cross-validation of single particle cryo-EM. Journal of Chemical
Information and Modeling, 60(5):2413–2418, 2020.
[18] Constantinos Daskalakis, Christos Tzamos, and Manolis Zampetakis. Ten steps of EM
suffice for mixtures of two gaussians. In Conference on Learning Theory, pages 704–710.
PMLR, 2017.
24

## Page 25

[19] Arthur P Dempster, Nan M Laird, and Donald B Rubin. Maximum likelihood from
incomplete data via the EM algorithm. Journal of the royal statistical society: series B
(methodological), 39(1):1–22, 1977.
[20] Rick Durrett. Probability: theory and examples, volume 49. Cambridge university press,
2019.
[21] Amitay Eldar, Keren Mor Waknin, Samuel Davenport, Tamir Bendory, Armin Schwartz-
man, and Yoel Shkolnisky.
Object detection under the linear subspace model with
application to cryo-EM images. arXiv preprint arXiv:2405.00364, 2024.
[22] Ayelet Heimowitz, Joakim And´en, and Amit Singer. APPLE picker: Automatic particle
picking, a low-effort cryo-EM framework. Journal of structural biology, 204(2):215–227,
2018.
[23] Richard Henderson. Avoiding the pitfalls of single particle cryo-electron microscopy:
Einstein from noise. Proceedings of the National Academy of Sciences, 110(45):18037–
18041, 2013.
[24] Richard Henderson, Andrej Sali, Matthew L Baker, Bridget Carragher, Batsal Devkota,
Kenneth H Downing, Edward H Egelman, Zukang Feng, Joachim Frank, Nikolaus Grig-
orieff, et al. Outcome of the first electron microscopy validation task force meeting.
Structure, 20(2):205–214, 2012.
[25] J Bernard Heymann. Validation of 3D EM reconstructions: The phantom in the noise.
AIMS biophysics, 2(1):21, 2015.
[26] Olav Kallenberg. Foundations of modern probability. Springer, 1997.
[27] Gerard J Kleywegt, Paul D Adams, Sarah J Butcher, Catherine L Lawson, Alexis
Rohou, Peter B Rosenthal, Sriram Subramaniam, Maya Topf, Sanja Abbott, Philip R
Baldwin, et al. Community recommendations on cryoEM data archiving and validation.
IUCrJ, 11(2), 2024.
[28] Theocharis Kyriacou, Guido Bugmann, and Stanislao Lauria. Vision-based urban nav-
igation procedures for verbally instructed robots. Robotics and Autonomous Systems,
51(1):69–80, 2005.
[29] Malcolm R Leadbetter, Georg Lindgren, and Holger Rootz´en. Extremes and related
properties of random sequences and processes.
Springer Science & Business Media,
2012.
[30] Yuhai Li, Jian Liu, Jinwen Tian, and Hongbo Xu. A fast rotated template matching
based on point feature. In MIPPR 2005: SAR and Multispectral Image Processing,
volume 6043, pages 453–459. SPIE, 2005.
[31] Sergio I Lopez and Leandro PR Pimentel. On the location of the maximum of a process:
L’evy, gaussian and multidimensional cases. arXiv preprint arXiv:1611.02334, 2016.
25

## Page 26

[32] Youdong Mao, Luis R Castillo-Menendez, and Joseph G Sodroski.
Reply to subra-
maniam, van heel, and henderson: Validity of the cryo-electron microscopy structures
of the HIV-1 envelope glycoprotein complex. Proceedings of the National Academy of
Sciences, 110(45):E4178–E4182, 2013.
[33] Youdong Mao, Liping Wang, Christopher Gu, Alon Herschhorn, Anik D´esormeaux,
Andr´es Finzi, Shi-Hua Xiang, and Joseph G Sodroski. Molecular architecture of the
uncleaved HIV-1 envelope glycoprotein trimer. Proceedings of the National Academy of
Sciences, 110(30):12438–12443, 2013.
[34] Amit Moscovich and Saharon Rosset. On the cross-validation bias due to unsupervised
preprocessing. Journal of the Royal Statistical Society Series B: Statistical Methodology,
84(4):1474–1502, 2022.
[35] Eva Nogales. The development of cryo-EM into a mainstream structural biology tech-
nique. Nature methods, 13(1):24–27, 2016.
[36] Alan V Oppenheim and Jae S Lim. The importance of phase in signals. Proceedings of
the IEEE, 69(5):529–541, 1981.
[37] Magda Peligrad and Wei Biao Wu.
Central limit theorem for fourier transforms of
stationary processes. 2010.
[38] Amelia Perry, Jonathan Weed, Afonso S Bandeira, Philippe Rigollet, and Amit Singer.
The sample complexity of multireference alignment. SIAM Journal on Mathematics of
Data Science, 1(3):497–517, 2019.
[39] Leandro PR Pimentel. On the location of the maximum of a continuous stochastic
process. Journal of Applied Probability, 51(1):152–161, 2014.
[40] Ali Punjani, John L Rubinstein, David J Fleet, and Marcus A Brubaker. cryoSPARC:
algorithms for rapid unsupervised cryo-EM structure determination. Nature methods,
14(3):290–296, 2017.
[41] Sjors HW Scheres. RELION: implementation of a bayesian approach to cryo-EM struc-
ture determination. Journal of structural biology, 180(3):519–530, 2012.
[42] Sjors HW Scheres. Semi-automated selection of cryo-EM particles in relion-1.3. Journal
of structural biology, 189(2):114–122, 2015.
[43] Vahid Shahverdi, Emanuel Str¨om, and Joakim And´en. Moment Constraints and Phase
Recovery for Multireference Alignment. arXiv preprint arXiv:2409.04868, 2024.
[44] Maxim Shatsky, Richard J Hall, Steven E Brenner, and Robert M Glaeser. A method
for the alignment of heterogeneous macromolecules from electron microscopy. Journal
of structural biology, 166(1):67–78, 2009.
[45] Yoav Shechtman, Yonina C Eldar, Oren Cohen, Henry Nicholas Chapman, Jianwei
Miao, and Mordechai Segev. Phase retrieval with application to optical imaging: a
contemporary overview. IEEE signal processing magazine, 32(3):87–109, 2015.
26

## Page 27

[46] Fred J Sigworth. A maximum-likelihood approach to single-particle image refinement.
Journal of structural biology, 122(3):328–339, 1998.
[47] Amit Singer.
Wilson statistics: derivation, generalization and applications to elec-
tron cryomicroscopy.
Acta Crystallographica Section A: Foundations and Advances,
77(5):472–479, 2021.
[48] Amit Singer and Fred J Sigworth. Computational methods for single-particle electron
cryomicroscopy. Annual review of biomedical data science, 3:163–190, 2020.
[49] E. Slutsky. ¨Uber stochastische Asymptoten und Grenzwerte. 1925.
[50] Carlos OS Sorzano, JL Vilas, Erney Ram´ırez-Aportela, J Krieger, D Del Hoyo, David
Herreros, Estrella Fernandez-Gim´enez, D March´an, JR Mac´ıas, I S´anchez, et al. Image
processing tools for the validation of CryoEM maps. Faraday Discussions, 240:210–227,
2022.
[51] Alex Stewart and Nikolaus Grigorieff. Noise bias in the refinement of structures derived
from single particles. Ultramicroscopy, 102(1):67–84, 2004.
[52] Sriram Subramaniam. Structure of trimeric HIV-1 envelope glycoproteins. Proceedings
of the National Academy of Sciences, 110(45):E4172–E4174, 2013.
[53] Itamar Talmi, Roey Mechrez, and Lihi Zelnik-Manor.
Template matching with de-
formable diversity similarity.
In Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, pages 175–183, 2017.
[54] Marin van Heel. Finding trimeric HIV-1 envelope glycoproteins in random noise. Pro-
ceedings of the National Academy of Sciences, 110(45):E4175–E4177, 2013.
[55] Shao-Hsuan Wang, Yi-Ching Yao, Wei-Hau Chang, and I-Ping Tu. Quantification of
model bias underlying the phenomenon of “Einstein from noise”.
Statistica Sinica,
31:2355–2379, 2021.
[56] Ji Xu, Daniel J Hsu, and Arian Maleki. Global analysis of expectation maximization
for mixtures of two gaussians. Advances in Neural Information Processing Systems, 29,
2016.
[57] Sheng Xu, Amnon Balanov, Amit Singer, and Tamir Bendory. Bayesian perspective
for orientation determination in cryo-EM with application to structural heterogeneity
analysis. bioRxiv, pages 2024–12, 2025.
[58] SH Y¨u.
Determination of absolute from relative X-ray intensity data.
Nature,
150(3796):151–152, 1942.
[59] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The
unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
27

## Page 28

[60] Andy Zhu. Multireference Alignment via Semidefinite Programming. Phd thesis, Prince-
ton University, 2013.
Appendix
Appendix organization.
Appendix A provides general preliminaries used throughout
the paper, including notation and common technical tools. Appendix B presents the aux-
iliary lemmas required for Theorem 4.1, whose full proof appears in Appendix B.5. For
Theorem 4.3, the necessary supporting results are given in Appendices C, with the proof
provided in Appendix D. Appendix E contains the proof of Proposition 5.1, establishing the
positive correlation property. Appendix F proves Theorem 5.2, which extends the results
to high-dimensional settings with i.i.d. noise that is not necessarily Gaussian. Finally, Ap-
pendix G provides the proof of Proposition 5.4, addressing the case of structured noise with
a circulant Gaussian covariance.
A
Preliminaries
Before we delve into the proofs, we fix notations and definitions and prove auxiliary results
that will be used in the proofs.
A.1
Notations
Recall the definitions of the Fourier transforms of x and ni from (2.6), and recall that the
signal length d is assumed to be even. Note that since x and ni are real-valued, their Fourier
coefficients satisfy the conjugate-symmetry relation:
X[k] = X[d −k],
Ni[k] = Ni[d −k].
(A.1)
In particular, |Ni[k]| = |Ni[d −k]| and ϕNi[k] = −ϕNi[d −k], which implies that only the first
d/2 + 1 components of N[k] are statistically independent.
The definition of the maximal correlation in (2.3) can be represented in the Fourier
domain as follows,
ˆRi ≜arg max
0≤r≤d−1
⟨ni, Trx⟩
(A.2)
= arg max
0≤r≤d−1
⟨F {ni} , F {Trx}⟩
(A.3)
= arg max
0≤r≤d−1
d−1
X
k=0
|X[k]| |Ni[k]| cos
2πkr
d
+ ϕNi[k] −ϕX[k]

.
(A.4)
To simplify notation, we define
Si[r] ≜
d−1
X
k=0
|X[k]| |Ni[k]| cos
2πkr
d
+ ϕNi[k] −ϕX[k]

,
(A.5)
28

## Page 29

for 0 ≤r ≤d −1, and therefore, ˆRi = arg max0≤r≤d−1 Si[r]. We note that for any 0 ≤i ≤
M −1, the random vector Si ≜(Si[0], Si[1], . . . , Si[d−1])T is Gaussian distributed, with zero
mean vector, and a circulant covariance matrix; therefore, it is a cyclo-stationary random
process.
Our goal is to investigate the phase and magnitude of the estimator ˆX in (2.7). Simple
manipulations reveal that, for any 0 ≤k ≤d −1, the estimator’s phases are given by,
ϕˆX[k] = ϕX[k] + arctan
 PM−1
i=0 |Ni[k]| sin (ϕe,i[k])
PM−1
i=0 |Ni[k]| cos (ϕe,i[k])
!
,
(A.6)
where we define,
ϕe,i[k] ≜2πkˆRi
d
+ ϕNi[k] −ϕX[k],
(A.7)
and
|ˆX[k]| = 1
M

M−1
X
i=0
|Ni[k]| ejϕe,i[k]
 .
(A.8)
A.2
The convergence of the Einstein from Noise estimator
Recall the definition of ϕe,i[k] in (A.7). Then, following (2.7), and simple algebraic manipu-
lation,
ˆX[k] = 1
M
M−1
X
i=0
|Ni[k]| ejϕNi[k]ej 2πk
d ˆRi
(A.9)
= ejϕX[k]
M
M−1
X
i=0
|Ni[k]| ejϕNi[k]ej 2πk
d ˆRie−jϕX[k]
(A.10)
= ejϕX[k]
M
M−1
X
i=0
|Ni[k]| ejϕe,i[k].
(A.11)
By applying the strong law of large numbers (SLLN) on the right-hand-side of (A.11), for
M →∞, we have,
ˆX[k]e−jϕX[k] = 1
M
M−1
X
i=0
|Ni[k]| ejϕe,i[k]
(A.12)
a.s.
−−→E [|N1[k]| cos (ϕe,1[k])] + jE [|N1[k]| sin (ϕe,1[k])] ,
(A.13)
where we have used the fact that the sequences of random variables {|Ni[k]| sin (ϕe,i[k])}M−1
i=0
and {|Ni[k]| cos (ϕe,i[k])}M−1
i=0
are i.i.d. with finite mean and variances.
We denote for every 0 ≤k ≤d −1:
µA,k ≜E [|N1[k]| sin(ϕe,1[k])] ,
(A.14)
29

## Page 30

µB,k ≜E [|N1[k]| cos(ϕe,1[k])] ,
(A.15)
the imaginary and real parts of the right-hand-side of (A.13), respectively. In addition, we
denote:
σ2
A,k ≜Var (|N1[k]| sin(ϕe,1[k])) ,
(A.16)
σ2
B,k ≜Var (|N1[k]| cos(ϕe,1[k])) .
(A.17)
In Theorem 4.1, we prove that µA,k = 0 while µB,k > 0. Consequently, by (A.13), as M →∞,
the EfN estimator converges to a non-vanishing signal, and its Fourier phases converge those
of the template (Einstein).
A.3
Conditioning on the Fourier frequency noise component
Throughout the proofs, we condition the noise realization Si (A.5) on the k-th Fourier co-
efficient Si|Ni[k], to capture the dependence of ˆRi on the noise component. Specifically, we
prove the following:
Lemma A.1. Recall the definition of Si (A.5). Then, for every k ∈

1, 2, . . . , d
2 −1, d
2 + 1, . . . d −1
	
,
Si|Ni[k] ∼N(µk,i, Σk,i),
(A.18)
where,
µk,i[r] ≜E [Si[r]|Ni[k]] = 2 |X[k]| |Ni[k]| cos
2πkr
d
+ ϕNi[k] −ϕX[k]

,
(A.19)
for 0 ≤r ≤d −1, and
Σk,i[r, s] ≜E [(Si[r] −ESi[r]) (Si[s] −ESi[s]) |Ni[k]]
= σ2
2
d−1
X
ℓ=0
|eXk[ℓ]|2 cos
2πℓ
d (r −s)

,
(A.20)
for 0 ≤r, s ≤d −1, where eXk is defined by:
eXk[ℓ] ≜





0
ifℓ= k, d −k,
X[ℓ]
ifℓ= 0, d/2,
√
2 · X[ℓ]
otherwise.
(A.21)
Remark A.2. In Lemma A.1, and throughout this work, we condition on Si|Ni[k] for all
k ̸= 0, d/2. Since the signals x and ni lie in Rd, their Fourier phases satisfy ϕX[0] = 0 and
ϕX[d/2] = 0. Therefore, we restrict our analysis to the convergence of the Fourier phases for
k ̸= 0, d/2, as the convergence at k = 0 and k = d/2 is trivial.
30

## Page 31

Note that the conditional process Si|Ni[k] is Gaussian because it is given by a linear
transform of i.i.d. Gaussian variables. Also, since its covariance matrix is circulant and
depends only on the difference between the two indices, i.e., Σk,i[r, s] = σk,i[|r −s|], it is
cycle-stationary with a cosine trend. The eigenvalues of this circulant matrix are given by
the DFT of its first row, and thus its ℓ-th eigenvalue equals |eXk[ℓ]|2, for 0 ≤ℓ≤d −1.
For simplicity of notation, whenever it is clear from the context, we will omit the depen-
dence of the above quantities on the i-th observation and k-th frequency indices, and we will
use µ[r] and Σ[r, s], instead. Furthermore, for convenience, we assume that the template
vector is normalized to unity, i.e. Pd−1
ℓ=0 |X[ℓ]|2 = 1.
Proof of Lemma A.1. By definition of Si (A.5), we have for every k ̸= 0, d/2,
Si [r] |Ni[k] =2 |X[k]| |Ni[k]| cos
2πkr
d
+ ϕNi[k] −ϕX[k]

+
X
ℓ̸=k,d−k
|X[ℓ]| |Ni[ℓ]| cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

,
(A.22)
where we have used the property of X[k] = X[d −k], Ni[k] = Ni[d −k], (A.1). Clearly, as
E [Ni [ℓ]] = 0, for every 0 ≤ℓ≤d −1, we have,
E

|X[ℓ]| |Ni[ℓ]| cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

= 0,
(A.23)
for every 0 ≤ℓ≤d −1. Combining (A.22) and (A.23) results,
µk,i[r] = E [Si[r]|Ni[k]] = 2 |X[k]| |Ni[k]| cos
2πkr
d
+ ϕNi[k] −ϕX[k]

,
(A.24)
proving the first result about the means.
The covariance term.
In the following, we derive the covariance term,
Σk,i[r, s] ≜E [(Si[r] −ESi[r]) (Si[s] −ESi[s]) |Ni[k]] .
(A.25)
Denote,
ρk,i [r] ≜Si[r] −ESi[r]
=
X
ℓ̸=k,d−k
|X[ℓ]| |Ni[ℓ]| cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

.
(A.26)
Denote the set
I = {1, 2, . . . k −1, k + 1, . . . , d/2 −1} ,
(A.27)
which defines the indices of the Fourier coefficients, excluding {0, k, d/2}.
31

## Page 32

As the sequences {|Ni[ℓ]|}d/2
ℓ=0 and {ϕNi[ℓ]}d/2
ℓ=0 satisfy Ni[ℓ] = Ni[d −ℓ], as well as X[ℓ] =
X[d −ℓ], we have,
ρk,i [r] =
X
ℓ̸=k,d−k
|X[ℓ]| |Ni[ℓ]| cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

=
=
X
ℓ∈{0,d/2}
|X[ℓ]| |Ni[ℓ]| cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

+ 2 ·
X
ℓ∈I
|X[ℓ]| |Ni[ℓ]| cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

,
(A.28)
where each one of the terms in the sum is independent.
Since the terms in the sum on the right-hand side of (A.28) are independent, that is,
E
h
Ni [ℓ1] Ni [ℓ2]
i
= E

|Ni [ℓ1]|2
δℓ1,ℓ2, it follows that,
Σk,i[r, s] = E [ρk,i [r] ρk,i [s] |Ni[k]]
= E


X
ℓ∈{0,d/2}
|X[ℓ]|2 |Ni[ℓ]|2 cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

cos
2πℓs
d
+ ϕNi[ℓ] −ϕX[ℓ]


+ 4 · E
"X
ℓ∈I
|X[ℓ]|2 |Ni[ℓ]|2 cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

cos
2πℓs
d
+ ϕNi[ℓ] −ϕX[ℓ]
#
.
(A.29)
The expectation value in (A.29) is composed of the multiplications of cosines. Applying
trigonometric identities, we obtain:
cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

cos
2πℓs
d
+ ϕNi[ℓ] −ϕX[ℓ]

= 1
2 cos
2πℓ(r −s)
d

+ 1
2 cos
2πℓ(r + s)
d
+ 2 (ϕNi[ℓ] −ϕX[ℓ])

.
(A.30)
For ℓ∈

1, . . . , d
2 −1
	
, the DFT coefficients Ni[ℓ] are i.i.d. circular complex Gaussian (un-
der white Gaussian noise in the time domain), hence independent across ℓ, with ϕNi[ℓ] ∼
Unif[−π, π) independent of |Ni[ℓ]|. Thus,
E

|Ni[ℓ]|2 cos
2πℓr
d
+ ϕNi[ℓ] −ϕX[ℓ]

cos
2πℓs
d
+ ϕNi[ℓ] −ϕX[ℓ]

= 1
2E

|Ni[ℓ]|2
cos
2πℓ(r −s)
d

= σ2
2 cos
2πℓ(r −s)
d

.
(A.31)
Substituting (A.31) into (A.29) leads to,
2
σ2E [ρk,i [r] ρk,i [s] |Ni[k]] =
X
ℓ∈{0,d/2}
|X[ℓ]|2 cos
2πℓ
d (r −s)

32

## Page 33

+ 4 ·
X
ℓ∈I
|X[ℓ]|2 cos
2πℓ
d (r −s)

.
(A.32)
As for every ℓ∈I, |X[ℓ]| = |X[d −ℓ]|, we have,
X
ℓ∈I
4 |X[ℓ]|2 cos
2πℓ
d (r −s)

=
X
ℓ̸={0,k,d/2,d−k}
2 |X[ℓ]|2 cos
2πℓ
d (r −s)

.
(A.33)
Substituting (A.33) into (A.32)
E [ρk,i [r] ρk,i [s] |Ni[k]] = σ2
2
d−1
X
ℓ=0
|eXk[ℓ]|2 cos
2πℓ
d (r −s)

,
for eXk[ℓ] defined in (A.21), which completes the proof.
A.4
Uniqueness of the maximizer
Lemma A.3 (Uniqueness of the maximizer). Recall the definition of Si from (A.5). Assume
d ≥6 is even. Fix k ∈{1, . . . , d
2 −1, d
2 + 1, . . . , d −1} and recall from Lemma A.1 that
Si|Ni[k] ∼N(µk,i, Σk,i),
Σk,i[r, s] = σ2
2
d−1
X
ℓ=0
eXk[ℓ]
2 cos
2πℓ
d (r −s)

.
(A.34)
Assume moreover that the template spectrum is non-vanishing, i.e. X[ℓ] ̸= 0 for all ℓ∈
{0, . . . , d −1}. Then, for every r ̸= s,
Var

(Si|Ni[k])r −(Si|Ni[k])s

= σ2
d−1
X
ℓ=0
eXk[ℓ]
2
1 −cos
  2πℓ
d (r −s)

> 0.
(A.35)
Consequently, the maximizer ˆRi = arg max0≤r≤d−1(Si|Ni[k])r is unique almost surely.
Proof of Lemma A.3. Fix r ̸= s and set m ≜r −s ̸≡0 (mod d). Using the covariance
formula and circulantness,
Var

(Si|Ni[k])r −(Si|Ni[k])s

= Σk,i[r, r] + Σk,i[s, s] −2Σk,i[r, s]
= σ2
d−1
X
ℓ=0
eXk[ℓ]
2
1 −cos
  2πℓ
d m

,
(A.36)
which proves the identity in (A.35). Each summand is nonnegative.
It remains to show strict positivity. We show that there is at least one term in the sum
that is strictly positive. Define
Hm ≜{ℓ∈{0, . . . , d −1} : ℓm ≡0 (mod d)}.
(A.37)
33

## Page 34

Then |Hm| = gcd(d, m) ≤d/2 since m ̸≡0 (mod d). Because d ≥6, we have
{0, . . . , d −1} \ (Hm ∪{k, d −k})
 ≥d −|Hm| −2
(A.38)
≥d −d
2 −2 = d
2 −2 > 0.
(A.39)
Hence we may choose ℓ0 ∈{0, . . . , d −1} \ (Hm ∪{k, d −k}). By construction, ℓ0 /∈Hm
implies ℓ0m ̸≡0 (mod d), i.e. cos(2πℓ0m/d) ̸= 1 and thus 1 −cos(2πℓ0m/d) > 0. Moreover,
since eXk[ℓ] = 0 only for ℓ∈{k, d −k} and X[ℓ] ̸= 0 for all ℓ, we have |eXk[ℓ0]|2 > 0. Therefore
the ℓ0-term in the sum is strictly positive, and the entire variance is strictly positive, proving
(A.35).
Finally, for any r ̸= s, the difference (Si|Ni[k])r−(Si|Ni[k])s is a non-degenerate Gaussian,
hence P
 (Si|Ni[k])r = (Si|Ni[k])s

= 0. A union bound over finitely many pairs implies ties
occur with probability 0, so the maximizer is unique almost surely.
A.5
Positive probability of each maximizer event
Lemma A.4 (Positive probability of each strict maximizer event). Recall the definition of
Si from (A.5). Assume d ≥6 is even. Fix k ∈{1, . . . , d
2 −1, d
2 +1, . . . , d−1} and recall from
Lemma A.1 that
Si|Ni[k] ∼N(µk,i, Σk,i),
Σk,i[r, s] = σ2
2
d−1
X
ℓ=0
eXk[ℓ]
2 cos
2πℓ
d (r −s)

.
(A.40)
Assume moreover that the template spectrum is non-vanishing, i.e. X[ℓ] ̸= 0 for all ℓ∈
{0, . . . , d −1}. Then, for every r ∈{0, . . . , d −1}, the event
Cr ≜
n
(Si|Ni[k])r > max
t̸=r (Si|Ni[k])t
o
(A.41)
has strictly positive probability:
P
 Cr
 Ni[k]

> 0,
(A.42)
for almost every realization of Ni[k].
Proof of Lemma A.4. Fix k as in the statement and condition on a realization of Ni[k]. Then
Y ≜Si|Ni[k] ∼N(m, Σ) with m = µk,i and Σ = Σk,i. By Lemma A.3, for every r ̸= s,
Var(Yr −Ys) > 0, hence Yr −Ys is a non-degenerate Gaussian and P(Yr = Ys) = 0. In
particular, ties occur with probability 0.
Since Σ is real, symmetric, and circulant, it is diagonalized by the DFT basis: there exist
eigenvectors {fℓ}d−1
ℓ=0 (the Fourier modes) such that the corresponding eigenvalues {λℓ}d−1
ℓ=0
are given by the DFT of the first row of Σ. In Model A.40, Lemma A.1 shows that these
eigenvalues satisfy
λℓ∝|eXk[ℓ]|2,
ℓ= 0, . . . , d −1.
(A.43)
34

## Page 35

Under the non-vanishing spectrum assumption X[ℓ] ̸= 0 for all ℓ, and by the definition (A.21),
we have
|eXk[ℓ]|2 > 0
for all ℓ/∈{k, d −k},
(A.44)
while |eXk[k]|2 = |eXk[d−k]|2 = 0. Hence Σ has exactly two zero eigenvalues, corresponding to
the ℓ= k and ℓ= d−k Fourier modes. In the real domain, these two modes are equivalently
represented by the cosine and sine vectors c, s ∈Rd
ct = cos

2πk
d t

,
st = sin

2πk
d t

,
(A.45)
for t = 0, . . . , d −1, so
ker(Σ) = span{c, s}.
(A.46)
Let L ≜range(Σ). Since Σ is symmetric, we have L = (ker Σ)⊥, and therefore the Gaussian
vector Y ∼N(m, Σ) is supported on the affine subspace m + L.
We fix r ∈{0, . . . , d −1} and consider the open cone
Cr ≜{y ∈Rd : yr > max
t̸=r yt}.
(A.47)
We claim that Cr ∩(m + L) ̸= ∅, i.e., the affine support contains at least one point whose
unique largest coordinate is the r-th.
To build such a point, we start from the r-th standard basis vector u = er, whose
maximizer is trivially at r (i.e., er ∈Cr), but which may fail to belong to m + L.
We
therefore remove from u its components along the two forbidden directions c and s (which
span ker(Σ) (A.46)). We define
α ≜⟨u, c⟩
⟨c, c⟩,
β ≜⟨u, s⟩
⟨s, s⟩,
v ≜u −α c −β s.
(A.48)
Because k /∈{0, d/2}, the sine and cosine vectors are orthogonal and have equal energy:
⟨c, s⟩= 0,
⟨c, c⟩= ⟨s, s⟩= d/2.
(A.49)
Hence ⟨v, c⟩= ⟨v, s⟩= 0, which means v ∈(ker Σ)⊥= L.
Next we show that v remains strongly peaked at the r-th coordinate. Since u = er, we
have ⟨u, c⟩= cr and ⟨u, s⟩= sr, and therefore
vr = 1 −
c2
r
⟨c, c⟩−
s2
r
⟨s, s⟩= 1 −c2
r + s2
r
d/2
= 1 −2
d,
(A.50)
where we used c2
r + s2
r = 1 (A.45). For t ̸= r, since u = er is the r-th standard basis vector,
we have ut = 0; thus, using (A.45) and (A.49), we obtain
vt = −α ct −β st = −crct + srst
d/2
= −2
d cos

2πk
d (t −r)

,
(A.51)
35

## Page 36

so |vt| ≤2/d for all t ̸= r. Consequently, letting
δ ≜vr −max
t̸=r vt,
(A.52)
we obtain the uniform lower bound for d > 4
δ ≥

1 −2
d

−2
d = 1 −4
d > 0.
(A.53)
Thus, within the direction v ∈L, the r-th coordinate exceeds every other coordinate by at
least δ.
Finally, because the affine support is m + L, any point of the form y(γ) = m + γv lies
in the support. The offset m may change the ordering for small γ, but scaling γ makes the
v-term dominate. In particular, let
M ≜max
t̸=r |mr −mt|.
(A.54)
Then for any t ̸= r,
yr(γ) −yt(γ) = (mr −mt) + γ(vr −vt) ≥−M + γδ.
(A.55)
Choosing γ > M/δ guarantees yr(γ) > yt(γ) for all t ̸= r, hence
y(γ) ∈Cr ∩(m + L),
(A.56)
proving Cr ∩(m + L) ̸= ∅.
Since Cr is open in Rd, the intersection Cr∩(m+L) contains a nonempty open subset of the
affine support m+L. A Gaussian measure assigns positive probability to any nonempty open
subset of its affine support [26]; therefore, P(Y ∈Cr) > 0, or equivalently P
 Cr
 Ni[k]

> 0,
completing the proof.
A.6
Auxiliary result for Proposition B.2
Let S(+) ∼N(µ, Σ) and S(−) ∼N(−µ, Σ) be two d-dimensional Gaussian vectors, where Σ
is a real, symmetric, circulant covariance matrix. Define the maximizers
ˆR(+) = arg max
0≤ℓ≤d−1
S(+)
ℓ
,
(A.57)
ˆR(−) = arg max
0≤ℓ≤d−1
S(−)
ℓ
.
(A.58)
and assume they are unique almost surely. We define the entries of µ as,
µℓ≜[µ]ℓ= cos
2πk
d ℓ+ φ

,
(A.59)
for φ ∈[0, 2π), and 0 ≤ℓ≤d −1. Note that −µℓ= cos
  2πk
d ℓ+ φ + π

, for 0 ≤ℓ≤d −1.
Then, we have the following result.
36

## Page 37

Proposition A.5. Consider the definitions in (A.59)–(A.58), and assume the maximizers
in (A.58) are unique a.s. Moreover, assume that for every r ∈{1, . . . , d −1},
P

S(+)
r
> max
t̸=r S(+)
t

> 0
and
P

S(−)
r
> max
t̸=r S(−)
t

> 0.
(A.60)
Fix 0 ≤ℓ≤d −1. If µℓ> 0, then,
P
h
ˆR(+) = ℓ
i
> P
h
ˆR(−) = ℓ
i
,
(A.61)
otherwise, if µℓ< 0, then,
P
h
ˆR(−) = ℓ
i
> P
h
ˆR(+) = ℓ
i
.
(A.62)
In particular, for any φ ∈[0, 2π) and 0 ≤k ≤d −1,
E

cos
2πk
d
ˆR(+) + φ

+ E

cos
2πk
d
ˆR(−) + φ + π

> 0.
(A.63)
Proof of Proposition A.5. By definition, it is clear that,
P
h
ˆR(+) = ℓ
i
= P

S(+)
ℓ
≥max
m̸=ℓS(+)
m

,
(A.64)
and,
P
h
ˆR(−) = ℓ
i
= P

S(−)
ℓ
≥max
m̸=ℓS(−)
m

,
(A.65)
for 0 ≤ℓ≤d −1. Since S(+) and S(−) can be decomposed as S(+) = Z + µ and S(−) = Z −µ,
where Z is a cyclo-stationary process, and µ is defined in (A.59). Then,
P

S(+)
ℓ
≥max
m̸=ℓS(+)
m

= P

Zℓ+ µℓ≥max
m̸=ℓZm + µm

,
(A.66)
and,
P

S(−)
ℓ
≥max
m̸=ℓS(−)
m

= P

Zℓ−µℓ≥max
m̸=ℓZm −µm

.
(A.67)
We will show that for any ℓsuch that µℓ> 0, we have,
P

Zℓ≥max
m̸=ℓ{Zm + µm −µℓ}

> P

Zℓ≥max
m̸=ℓ{Zm −µm + µℓ}

,
(A.68)
which in turn implies that P{ˆR(+) = ℓ} > P{ˆR(−) = ℓ}.
By definition, since Z is a zero-mean Gaussian, cyclo-stationary random process (i.e.,
with a real, symmetric, circulant covariance matrix), its cumulative distribution function FZ
is invariant under cyclic shifts, i.e.,
FZ (z0, z1, . . . , zd−1) = FZ (zτ, zτ+1, . . . , zτ+d−1) ,
(A.69)
37

## Page 38

for any τ ∈Z, with indices taken modulo d. Moreover, reversing the time indices does not
affect the distribution; that is,
FZ (z0, z1, . . . , zℓ−1, zℓ, zℓ+1, . . . , zd−1) = FZ (zd−1, zd−2, . . . , zℓ+1, zℓ, zℓ−1, ..., z0) .
(A.70)
Combining (A.69) and (A.70) yields,
FZ (zℓ, zℓ+1, zℓ+2, . . . , zℓ−2, zℓ−1) = FZ (zℓ, zℓ−1, zℓ−2, . . . , zℓ+2, zℓ+1) .
(A.71)
Accordingly, let us define the Gaussian vectors Z(1) and Z(2), such that their m-th entry is,
[Z(1)]m = Zℓ+m −Zℓ,
(A.72)
[Z(2)]m = Zℓ−m −Zℓ,
(A.73)
for 1 ≤m ≤d −1. It is clear from (A.71) that Z(1) and Z(2) have the same cumulative
distribution function, i.e.,
FZ(1) = FZ(2).
(A.74)
Therefore, the following holds,
P

Zℓ≥max
m̸=0 {Zℓ+m + µℓ+m −µℓ}

= P

0 ≥max
m̸=0 {Zℓ+m −Zℓ+ µℓ+m −µℓ}

= P

0 ≥max
m̸=0 {Zℓ−m −Zℓ+ µℓ+m −µℓ}

= P

Zℓ≥max
m̸=0 {Zℓ−m + µℓ+m −µℓ}

,
(A.75)
where the second equality follows from (A.74). Next, we note that for every 0 < m ≤d −1
and µℓ> 0,
µℓ−m + µℓ+m = 2µℓcos
2πk
d m

.
(A.76)
Therefore,
µℓ−µℓ−m + µℓ−µℓ+m = 2µℓ

1 −cos
2πk
d m

≥0,
(A.77)
which implies
µℓ−µℓ−m ≥µℓ+m −µℓ,
(A.78)
or, equivalently,
µℓ−µℓ+m ≥µℓ−m −µℓ.
(A.79)
38

## Page 39

Remark A.6. According to (A.77), the inequalities in (A.78) and (A.79) are strict whenever
cos
  2πk
d m

< 1, which holds for the majority of values of m. In particular, at least d/2 of
the inequalities are strict for 0 ≤m ≤d −1.
Following from (A.78), (A.79), and the last remark, we have the following auxiliary
Lemma, which we prove below.
Lemma A.7. Assume the maximizers in (A.58) are unique a.s. and satisfy (A.60). Then,
for µℓ> 0, we have,
P

Zℓ≥max
m̸=0 {Zℓ+m + µℓ+m −µℓ}

> P

Zℓ≥max
m̸=0 {Zℓ−m −µℓ−m + µℓ}

.
(A.80)
Note that (A.80) is equivalent to the following expression, by a change of index notation:
P

Zℓ≥max
m̸=ℓ{Zm + µm −µℓ}

> P

Zℓ≥max
m̸=ℓ{Zm −µm + µℓ}

,
(A.81)
which proves (A.68). A similar result can be obtained for the case where µℓ< 0, i.e.,
P

Zℓ≥max
m̸=ℓ{Zm + µm −µℓ}

< P

Zℓ≥max
m̸=ℓ{Zm −µm + µℓ}

,
(A.82)
which completes the proofs of (A.61)–(A.62).
Finally, we prove (A.63). By definition, it is clear that
E

cos
2πk
d
ˆR(+) + φ

+ E

cos
2πk
d
ˆR(−) + φ + π

=
d−1
X
ℓ=0
cos
2πk
d ℓ+ φ
 h
P

ˆR(+) = ℓ

−P

ˆR(−) = ℓ
i
,
(A.83)
where we have used the fact that cos(α + π) = −cos α, for any α ∈R.
By (A.61)–(A.62), as for any 0 ≤ℓ≤d−1 such that µℓ= cos
  2πk
d ℓ+ φ

> 0 it holds that
P[ˆR(+) = ℓ] > P[ˆR(−) = ℓ], otherwise, for 0 ≤ℓ≤d −1 such that µℓ= cos
  2πk
d ℓ+ φ

< 0, it
holds that P[ˆR(+) = ℓ] < P[ˆR(−) = ℓ]. Therefore,
d−1
X
ℓ=0
cos
2πk
d ℓ+ φ
 h
P

ˆR(+) = ℓ

−P

ˆR(−) = ℓ
i
> 0,
(A.84)
which in light of (A.83) concludes the proof.
It is left to prove Lemma A.7.
Proof of Lemma A.7. Using (A.78) and (A.79), we obtain the following inequalities for µℓ>
0,
max
m̸=0 {Zℓ−m −µℓ+m + µℓ} ≥max
m̸=0 {Zℓ−m + µℓ−m −µℓ} ,
(A.85)
39

## Page 40

and
max
m̸=0 {Zℓ−m −µℓ−m + µℓ} ≥max
m̸=0 {Zℓ−m + µℓ+m −µℓ} ,
(A.86)
As a result, we also have the following probabilistic inequalities,
P

Zℓ< max
m̸=0 {Zℓ−m −µℓ+m + µℓ}

≥P

Zℓ< max
m̸=0 {Zℓ−m + µℓ−m −µℓ}

,
(A.87)
and,
P

Zℓ< max
m̸=0 {Zℓ−m −µℓ−m + µℓ}

≥P

Zℓ< max
m̸=0 {Zℓ−m + µℓ+m −µℓ}

.
(A.88)
Next, we show that these probabilistic inequalities are in fact strict. Define the set of indices
where the inequality in (A.79) is strict,
M = {m : µℓ−µℓ+m > µℓ−m −µℓ} .
(A.89)
From Remark A.6, we know that |M| ≥d/2. Now define the event,
Cr =

Zℓ−r −µℓ−r + µℓ= max
m̸=0 {Zℓ−m −µℓ−m + µℓ}

,
(A.90)
i.e., the event that Zℓ−r −µℓ−r + µℓattains the maximum in the expression above.
By
assumption (A.60), the event Cr for every r, i.e. P(Cr) > 0. Thus, we may choose r ∈M
with P(Cr) > 0, i.e.,
r ∈M,
and
P(Cr) = P

max
m̸=0
n
Zℓ−m −µℓ−m + µℓ
o
= Zℓ−r −µℓ−r + µℓ

> 0.
(A.91)
Then, by the law of total probability,
P

Zℓ< max
m̸=0 {Zℓ−m −µℓ−m + µℓ}

= P

Zℓ< max
m̸=0 {Zℓ−m −µℓ−m + µℓ} | Cr

P [Cr]
+ P

Zℓ< max
m̸=0 {Zℓ−m −µℓ−m + µℓ} | Cc
r

P [Cc
r] .
(A.92)
From (A.88), we have,
P

Zℓ< max
m̸=0 {Zℓ−m −µℓ−m + µℓ} | Cc
r

≥P

Zℓ< max
m̸=0 {Zℓ−m + µℓ+m −µℓ} | Cc
r

. (A.93)
Additionally, since r ∈M, it follows that,
P

Zℓ< max
m̸=0 {Zℓ−m −µℓ−m + µℓ} | Cr

> P

Zℓ< max
m̸=0 {Zℓ−m + µℓ+m −µℓ} | Cr

. (A.94)
40

## Page 41

Substituting (A.93) and (A.94) into (A.92) yields,
P

Zℓ< max
m̸=0 {Zℓ−m −µℓ−m + µℓ}

> P

Zℓ< max
m̸=0 {Zℓ−m + µℓ+m −µℓ} | Cr

P [Cr]
+ P

Zℓ< max
m̸=0 {Zℓ−m + µℓ+m −µℓ} | Cc
r

P [Cc
r] .
(A.95)
By the law of total probability, the right-hand-side of (A.95) is,
P

Zℓ< max
m̸=0 {Zℓ−m + µℓ+m −µℓ}

= P

Zℓ< max
m̸=0 {Zℓ−m + µℓ+m −µℓ} | Cr

P [Cr]
+ P

Zℓ< max
m̸=0 {Zℓ−m + µℓ+m −µℓ} | Cc
r

P [Cc
r] .
(A.96)
Combining (A.95) and (A.96), we conclude,
P

Zℓ< max
m̸=0 {Zℓ−m −µℓ−m + µℓ}

> P

Zℓ< max
m̸=0 {Zℓ−m + µℓ+m −µℓ}

.
(A.97)
Equivalently, we can express (A.97) as a complementary event, and obtain,
P

Zℓ≥max
m̸=0 {Zℓ+m + µℓ+m −µℓ}

> P

Zℓ≥max
m̸=0 {Zℓ−m −µℓ−m + µℓ}

,
(A.98)
which proves (A.80), and completes the proof.
B
Proof of Theorem 4.1
First, we prove several auxiliary statements needed in the proof of Theorem 4.1. Recall the
definition of ϕe,i[k] in (A.7) and of µA,k, µB,k, σ2
A,k, σ2
B,k in (A.14)–(A.17).
Notation for convergence rate of the Fourier phases.
Denote for every 0 ≤k ≤d−1,
AM,k ≜
1
√
M
M−1
X
i=0
|Ni[k]| sin (ϕe,i[k]) ,
(B.1)
and,
BM,k ≜1
M
M−1
X
i=0
|Ni[k]| cos (ϕe,i[k]) .
(B.2)
Note that AM,k is the imaginary part of the EfN estimator, multiplied by the phase of the
template signal as defined in (A.11), but is normalized by 1/
√
M instead of 1/M, to facilitate
the analysis of the convergence rate. Similarly, BM,k corresponds to the real part in (A.11).
Additionally, we define the following Gaussian random variable Qk
Qk ∼N
 
0, σ2
A,k
µ2
B,k
!
,
(B.3)
for every 0 ≤k ≤d −1.
41

## Page 42

The main results of this section.
Recall that by the SLLN (A.13), the EfN estimator
converges to,
ˆX[k]
a.s.
−−→ejϕX[k] · E [|N1[k]| cos (ϕe,1[k])] + jE [|N1[k]| sin (ϕe,1[k])]
(B.4)
= ejϕX[k] (µB,k + jµA,k) .
(B.5)
In Sections B.1 and B.2, we prove that µA,k = 0 and µB,k > 0. These results, combined with
(B.5), imply that the EfN estimator converges to a non-vanishing signal, and its Fourier
phases converge those of Einstein as M →∞. In Sections B.3 and B.4, we analyze the
convergence rate of the Fourier phases, first establishing convergence rate in distribution to
Qk, and then proving convergence rate in expectation.
B.1
Convergence of the Fourier phases
Lemma B.1 (Convergence of the Fourier phases). Recall the definition of ϕe,i[k] in (A.7).
Then we have,
µA,k ≜E [|N1[k]| sin(ϕe,1[k])] = 0,
(B.6)
for every 0 ≤k ≤d −1.
Proof of Lemma B.1. Let D[k] ≜ϕX[k] −ϕN1[k], and recall the definition of ˆRi in (A.4):
ˆRi = arg max
0≤r≤d−1
d−1
X
k=0
|X[k]| |Ni[k]| cos
2πkr
d
+ ϕNi[k] −ϕX[k]

.
(B.7)
Note that ˆRi is a function of
ˆRi = ˆRi

{|Ni[k]|}d−1
k=0 , {|X[k]|}d−1
k=0 , {ϕNi[k]}d−1
k=0 , {ϕX[k]}d−1
k=0

,
(B.8)
and it depends on ϕNi[k] and ϕX[k] only through D[k]. Accordingly, viewing ˆR1 as a function
of D[k], for fixed {|Ni[k]|}d−1
k=0 , {|X[k]|}d−1
k=0, we have,
ˆR1 (−D[0], −D[1], . . . , −D[d −1]) = −ˆR1 (D[0], D[1], . . . , D[d −1]) .
(B.9)
Namely, from symmetry arguments, by flipping the signs of all the phases, the location of
the maximum flips its sign as well. Then, by the law of total expectation,
µA,k = E

|N1[k]| sin
2πk
d
ˆR1 + ϕN1[k] −ϕX[k]

= E

|N1[k]| · E

sin
2πk
d
ˆR1 + ϕN1[k] −ϕX[k]
 {|N1[k]|}d−1
k=0

.
(B.10)
The inner expectation in (B.10) is taken w.r.t. the uniform distribution randomness of the
phases {ϕN1[k]}d−1
k=0 ∈[−π, π). However, due to (B.9), and since the sine function is odd
around zero, the integration in (B.10) nullifies. Therefore,
E

sin
2πk
d
ˆR1 + ϕN1[k] −ϕX[k]
 {|N1[k]|}d−1
k=0

= 0,
(B.11)
and thus µA,k = 0.
42

## Page 43

B.2
Convergence to non-vanishing signal
Proposition B.2 (Convergence to non-vanishing signal). Recall the definition of ϕe,i[k] in
(A.7).
Fix d ∈N, and assume that X[k] ̸= 0 for all 0 < k ≤d −1.
Then, for any
0 ≤k ≤d −1,
µB,k ≜E[|N1[k]| cos(ϕe,1[k])] > 0.
(B.12)
Proof of Proposition B.2. By the law of total expectation, we have,
E[|N1[k]| cos(ϕe,1[k])] = E [|N1[k]| · E (cos(ϕe,1[k])| N1[k])]
= E
"
|N1[k]| · E
 
cos
 
2πkˆR1
d
+ ϕN1[k] −ϕX[k]
! N1[k]
!#
.
(B.13)
More explicitly, we can write,
E[|N1[k]| cos(ϕe,1[k])] =
1
2π
Z ∞
0
dnnf|N1[k]|(n)
Z π
−π
dφE

cos
2πk
d
ˆR1 + φ
 |N1[k]| = n, ϕN1[k] = ϕX[k] + φ

.
(B.14)
Now, note that the inner integral can be written as,
Z π
−π
dφE

cos
2πk
d
ˆR1 + φ
 |N1[k]| = n, ϕN1[k] = ϕX[k] + φ

=
Z π
0
dφ E

cos
2πk
d
ˆR1 + φ
 |N1[k]| = n, ϕN1[k] = ϕX[k] + φ

+
+
Z π
0
dφ E

cos
2πk
d
ˆR1 + φ + π
 |N1[k]| = n, ϕN1[k] = ϕX[k] + φ + π

.
(B.15)
Now, we apply Proposition A.5 on the integrands in (B.15). Using its notation, we define
the Gaussian process:
S(+) = S1|N1[k],
(B.16)
where the right-hand side is defined as in (A.18). By (A.19), the mean vector of S1|N1[k] has
a cosine trend, as assumed in Proposition A.5 in (A.59). Additionally, S1|N1[k] is a Gaussian
cyclo-stationary process, as described in (A.20). The final condition to verify is (A.60), which
is satisfied by Lemma A.4.
Since the conditional distribution of ˆR1 given {|N1[k]| = n, ϕN1[k] = ϕX[k] + φ} matches
that of ˆR(+) in (A.57), and similarly, given {|N1[k]| = n, ϕN1[k] = ϕX[k] + φ + π}, it matches
ˆR(−) in (A.58), the sum of the integrands on the right-hand side of (B.15) equals the left-
hand side of (A.63). By Proposition A.5, this sum is positive for all φ ∈[0, π]. Together
with (B.14), this completes the proof of Proposition B.2.
43

## Page 44

B.3
Convergence rate in distribution of the Fourier phases
Proposition B.3 (Convergence in distribution of the Fourier phases). Fix d ∈N. Then,
for any 0 ≤k ≤d −1,
√
M · tan (ϕˆX[k] −ϕX[k])
D−→Qk,
(B.17)
as M →∞, where Qk is defined in (B.3).
Proof of Proposition B.3. Recall the definition of AM,k, BM,k in (B.1), (B.2), of Qk in (B.3),
and of ϕe,i[k] in (A.7). Then, following (A.6), the left-hand-side of (B.17) is given by,
√
M · tan [ϕˆX[k] −ϕX[k]] =
1
√
M
PM−1
i=0 |Ni[k]| sin (ϕe,i[k])
1
M
PM−1
i=0 |Ni[k]| cos (ϕe,i[k])
≜AM,k
BM,k
.
(B.18)
Since {Ni}M−1
i=0
is an i.i.d. sequence of random variables, and because each ϕe,i depends on
Ni solely (in particular, independent of Nj, for j ̸= i), we have that {|Ni[k]| sin (ϕe,i[k])}M−1
i=0
and {|Ni[k]| cos (ϕe,i[k])}M−1
i=0
are two sequences of i.i.d. random variables. Recall the defini-
tion of µA,k, and σ2
A,k, in (A.14), (A.16), respectively:
µA,k ≜E [|N1[k]| sin(ϕe,1[k])] ,
(B.19)
σ2
A,k ≜Var (|N1[k]| sin(ϕe,1[k])) ,
(B.20)
which are the mean value and variance of AM,k, as defined in (B.1). Then, by the CLT:

AM,k −
√
MµA,k

D−→Ak,
(B.21)
where Ak ∼N(0, σ2
A,k). In particular, by Lemma B.1, µA,k = 0.
Next, we analyze the denominator in (B.18). Specifically, we already saw that {|Ni[k]| cos (ϕe,i[k])}M−1
i=0
form a sequence of i.i.d. random variables, and thus by the SLLN we have BM,k
a.s.
−−→µB,k,
where,
µB,k ≜E [|N1[k]| cos(ϕe,1[k])] .
(B.22)
By Proposition B.2, µB,k > 0. Thus, applying Slutsky’s Theorem on the ratio
AM,k
BM,k , we
obtain,
AM,k
BM,k
D−→N
 
0, σ2
A,k
µ2
B,k
!
= Qk,
(B.23)
which concludes the proof.
44

## Page 45

B.4
Convergence rate in expectation of the Fourier phases
Proposition B.4 (Convergence rate of the Fourier phases). Recall the definitions of µB,k,
and σ2
A,k in (A.15), and (A.17), respectively. Assume that X[k] ̸= 0, for all 0 < k ≤d −1.
Then, as M →∞,
lim
M→∞
E|ϕˆX[k] −ϕX[k]|2
1/M
= σ2
A,k
µ2
B,k
.
(B.24)
Proof of Proposition B.4. Recall the definitions of AM,k and BM,k in (B.1) and (B.2), respec-
tively, and of Qk in (B.3). Then, using the phase difference expression in (A.6), it follows
that establishing (B.24) is equivalent to proving the following:
lim
M→∞
E
h
arctan2 
1
√
M
AM,k
BM,k
i
1
M E [Q2
k]
= 1,
(B.25)
for every 0 ≤k ≤d −1. Recall by the definition of Qk in (B.3) that E [Q2
k] = σ2
A,k/µ2
B,k,
which is equivalent to the right-hand-side of (B.24).
For brevity, we fix k, and denote AM = AM,k, BM = BM,k, µB = µB,k, σ2
A = σ2
A,k. Using
(A.6) it is clear that,
√
M · tan [ϕˆX[k] −ϕX[k]] =
1
√
M
PM−1
i=0 |Ni[k]| sin (ϕe,i[k])
1
M
PM−1
i=0 |Ni[k]| cos (ϕe,i[k])
≜AM
BM
,
(B.26)
It is important to note that the denominator BM can be zero with positive probability,
implying that the expression in (B.26) may diverge with non-zero probability. Therefore, it
is necessary to control the occurrence of such events. To this end, BM
a.s.
−−→µB, by SLLN (see
Section A.2), where µB is defined in (A.15). Fix 0 < ϵ < µB, and proceed by decomposing
as follows:
E

arctan2
 1
√
M
AM
BM

= E

arctan2
 1
√
M
AM
BM

1|BM|>ϵ

+ E

arctan2
 1
√
M
AM
BM

1|BM|<ϵ

.
(B.27)
The next lemma shows that the second term at the r.h.s. of (B.27) converges to zero
with rate O(1/M 2).
Lemma B.5. The following inequality holds,
E

arctan2
 1
√
M
AM
BM

1|BM|<ϵ

≤D
M 2,
(B.28)
for a finite D > 0.
In addition, we have the following asymptotic relation for the last term in (B.27).
45

## Page 46

Lemma B.6. The following asymptotic relation hold,
lim
M→∞
E
h
arctan2 
1
√
M
AM
BM

1|BM|>ϵ
i
1
M E [Q2
k]
= 1.
(B.29)
We prove these lemmas below. Substituting (B.28) and (B.29) in (B.27), leads to (B.25),
and completing the proof of the proposition.
Proof of Lemma B.5. Since arctan(x) ≤π
2, for any x ∈R, we have
E

arctan2
 1
√
M
AM
BM

1|BM|<ϵ

≤π2
4 · E

1|BM|<ϵ

(B.30)
≤π2
4 · P (BM < ϵ)
(B.31)
= π2
4 · P (BM −µB < ϵ −µB)
(B.32)
≤π2
4 · P (|BM −µB| > µB −ϵ) .
(B.33)
Let us denote the summand in the denominator in (B.26) by Vi ≜|Ni[k]| cos (ϕe,i[k]), for
0 ≤i ≤M −1. Then, we note that,
E(V 4
i ) = E [|Ni[k]| cos (ϕe,i[k])]4 ≤E [|Ni[k]|]4 < ∞.
(B.34)
Thus, by Chebyshev’s inequality,
P (|BM −µB| > µB −ϵ) ≤E [BM −µB]4
(µB −ϵ)4
.
(B.35)
Now, by the definition of BM, we have,
E [BM −µB]4 =
1
M 4
M−1
X
i,j,k,l=0
E [(Vi −µB) (Vj −µB) (Vk −µB) (Vl −µB)]
(B.36)
=
1
M 4
h
M · E [V1 −µB]4 + 3M(M −1)
 E [V1 −µB]22i
.
(B.37)
Therefore, it is evident that there exists a constant D1, which depends on the second and
fourth moments of V1, such that,
E [BM −µB]4
(µB −ϵ)4
≤
D1
(µB −ϵ)4 M 2.
(B.38)
Thus, plugging (B.35) and (B.38) into (B.33) leads to,
E

arctan2
 1
√
M
AM
BM

1|BM|<ϵ

≤π2
4 ·
D1
(µB −ϵ)4 M 2.
(B.39)
Thus, the second term at the r.h.s. of (B.27) indeed converges to zero as 1/M 2.
46

## Page 47

Proof of Lemma B.6. We analyze the first term at the r.h.s. of (B.27). We will show that,
1 ≤lim
M→∞
E
h
arctan2 
1
√
M
AM
BM

1|BM|>ϵ
i
1
M E [Q2
k]
≤µ2
B
ϵ2 .
(B.40)
As this is true for every ϵ < µB, it would imply that,
lim
M→∞
E
h
arctan2 
1
√
M
AM
BM

1|BM|>ϵ
i
1
M E [Q2
k]
= 1.
(B.41)
First, due to the monotonicity of arctan2 (·),
E

arctan2
 1
√
M
AM
BM

1|BM|>ϵ

≤E

arctan2
 AM
ϵ
√
M

.
(B.42)
We decompose the right-hand-side of (B.42) into two events, as follows,
E

arctan2
 AM
ϵ
√
M

= E

arctan2
 AM
ϵ
√
M

1|AM|>ϵ
√
M

+ E

arctan2
 AM
ϵ
√
M

1|AM|<ϵ
√
M

.
(B.43)
By the SLLN, AM/
√
M
a.s.
−−→µA (see Section A.2), where µA = 0 (Lemma B.1). In addition,
by Proposition B.3, we have,
AM
D−→N
 0, σ2
A

,
(B.44)
by the CLT. Then, by arguments similar to those used in Lemma B.5, the first term on the
right-hand side of (B.43) satisfies:
E

arctan2
 AM
ϵ
√
M

1|AM|>ϵ
√
M

≤˜D/M 2.
(B.45)
Namely, the first term at the r.h.s. of (B.43) converges to zero with rate O( 1
M2).
For the last term in the right-hand-side of (B.43), we prove the following:
lim
M→∞
E
h
arctan2 
AM
ϵ
√
M

1|AM|<ϵ
√
M
i
1
M E
h  AM
ϵ
21|AM|<ϵ
√
M
i
= 1.
(B.46)
Since [arctan(x)]2/x2 →1 as x →0, it follows that the Taylor expansion of arctan(x) around
x = 0, which holds for |x| < 1, and is applicable on the event
n
AM < ϵ
√
M
o
:
E

arctan2
 AM
ϵ
√
M

1|AM|<ϵ
√
M

= E


∞
X
k=0
(−1)k
h
1
√
M
AM
ϵ
i2k+1
2k + 1
1|AM|<ϵ
√
M


2
.
(B.47)
47

## Page 48

The right-hand-side of (B.47) can be decomposed to,
E


∞
X
k=0
(−1)k
h
1
√
M
AM
ϵ
i2k+1
2k + 1
1|AM|<ϵ
√
M


2
= 1
M E
"AM
ϵ
2
1|AM|<ϵ
√
M
#
+
X
(k1,k2)̸=(0,0)
(−1)k1+k2
(2k1 + 1) (2k2 + 1)E
" AM
√
Mϵ
(2k1+2k2+2)
1|AM|<ϵ
√
M
#
.
(B.48)
Now, since the term at the left-hand-side of (B.48) as well as the first term at the right-
hand-side of (B.48), are bounded for every M and converges to zero, then also the last term
at the right-hand-side of (B.48) is bounded for every M and converge to zero as M →∞.
Specifically, we note that the last term converges to zero with rate 1/M 2, while the first term
in the right-hand-side converges to zero with rate 1/M. Thus, (B.46) is satisfied. Finally,
we have,
lim
M→∞
1
M E
h  AM
ϵ
21|AM|<ϵ
√
M
i
1
M E
h  AM
ϵ
2i
= 1 −lim
M→∞
1
M E
h  AM
ϵ
21|AM|>ϵ
√
M
i
1
M E
h  AM
ϵ
2i
.
(B.49)
As the probability of the event
n
|AM| > ϵ
√
M
o
is O (1/M 2), it follows that,
lim
M→∞
1
M E
h  AM
ϵ
21|AM|>ϵ
√
M
i
1
M E
h  AM
ϵ
2i
= 0.
(B.50)
Then, following (B.49),(B.50), we have,
lim
M→∞
1
M E
h  AM
ϵ
21|AM|<ϵ
√
M
i
1
M E
h  AM
ϵ
2i
= 1.
(B.51)
By definition σ2
A = E [A2
M]. Therefore substitution (B.51) into (B.46) leads to
lim
M→∞
E
h
arctan2 
AM
ϵ
√
M

1|AM|<ϵ
√
M
i
1
M
σ2
A
ϵ2
= 1.
(B.52)
Substituting (B.45), and (B.52) into (B.43) results,
lim
M→∞
E
h
arctan2 
1
√
M
AM
ϵ

1|BM|>ϵ
i
1
M E [Q2
k]
= µ2
B
ϵ2 ,
(B.53)
where E [Q2
k] = σ2
A/µ2
B. Then, substituting (B.53) into (B.42) results,
lim
M→∞
E
h
arctan2 
1
√
M
AM
BM

1|BM|>ϵ
i
1
M E [Q2
k]
≤µ2
B
ϵ2 .
(B.54)
48

## Page 49

which proves the upper bound in (B.40).
Similarly, since BM
a.s.
−−→µB, for any ϵ2 > 0, we have,
lim
M→∞E
"AM
BM
2
1{BM>ϵ}
#
≥lim
M→∞E
"AM
BM
2
1{BM>ϵ}∧{BM<µB+ϵ2}
#
(B.55)
≥
σ2
A
(µB + ϵ2)2.
(B.56)
Since (B.56) is true for every ϵ2 > 0, we get the lower bound in (B.40), which concludes the
proof of (B.41).
B.5
Proof of Theorem 4.1
Convergence of the Fourier magnitudes.
We start with the convergence of the es-
timator’s magnitudes.
Recall the definition of ϕe,i[k] in (A.7).
According to (A.13), we
have,
ˆX[k]e−jϕX[k]
a.s.
−−→
E [|N1[k]| cos (ϕe,1[k])] + jE [|N1[k]| sin (ϕe,1[k])]
,
(B.57)
Clearly,
e−jϕX[k] = 1. By Lemma B.1,
µA,k = E [|N1[k]| sin (ϕe,1[k])] = 0.
(B.58)
By Proposition B.2,
µB,k = E [|N1[k]| cos (ϕe,1[k])] > 0.
(B.59)
Combining (B.57), (B.58), and (B.59) proves the convergence of the estimator’s magnitudes
of (4.3).
Convergence of the Fourier phases.
Next, we prove the Fourier phases convergence of
Theorem 4.1, starting with (4.1). To this end, recall (A.6)
ϕˆX[k] −ϕX[k] = arctan
 PM−1
i=0 |Ni[k]| sin (ϕe,i[k])
PM−1
i=0 |Ni[k]| cos (ϕe,i[k])
!
,
(B.60)
Using the continuous mapping theorem, it is evident that it suffices to prove that,
PM−1
i=0 |Ni[k]| sin (ϕe,i[k])
PM−1
i=0 |Ni[k]| cos (ϕe,i[k])
a.s.
−−→0.
(B.61)
This, however, follows by applying the SLLN,
PM−1
i=0 |Ni[k]| sin (ϕe,i[k])
PM−1
i=0 |Ni[k]| cos (ϕe,i[k])
a.s.
−−→µA,k
µB,k
,
(B.62)
49

## Page 50

where µA,k ≜E [|N1[k]| sin(ϕe,1[k])] and µB,k ≜E [|N1[k]| cos(ϕe,1[k])], defined in (A.14), and
(A.15), respectively.
By Lemma B.1, µA,k = 0, while by Proposition B.2, we have that
µB,k > 0, and thus their ratio converges a.s. to zero by the continuous mapping theorem.
Thus, we proved that ϕˆX[k]
a.s.
−−→ϕX[k].
Finally, we prove the convergence rate, given in (4.2). According to Proposition B.4, we
have,
lim
M→∞
E|ϕˆX[k] −ϕX[k]|2
1/M
= σ2
A,k
µ2
B,k
,
(B.63)
which completes the proof of the Theorem.
Remark B.7. Note that the above result implies that Ck in (4.2) is given by,
Ck ≜σ2
A,k
µ2
B,k
= E
 [|N1[k]| sin(ϕe,1[k])]2
(E[|N1[k]| cos(ϕe,1[k])])2 .
(B.64)
C
High-dimensional argmax asymptotics
In this section, we present a key proposition that plays a central role in the proof of Theo-
rem 4.3.
Proposition C.1 (High-dimensional argmax asymptotics). Let S ∼N(µ, Σ) be a d-dimensional
Gaussian random vector, with mean µ and a covariance matrix Σ. Assume that |Σij| =
ρ|i−j|, where {ρℓ}ℓ∈N is a sequence of real-valued numbers such that ρ0 = 1, ρℓ< 1, and
ρℓlog ℓ→0, as ℓ→∞.
Assume also that √log d · max1≤i≤d |µi| →0, as d →∞,
and let ˆR ≜arg max {S0, S1, . . . , Sd−1}.
Then, for a bounded deterministic function f :
{0, 1, . . . , d −1} →R, we have,
lim
d→∞E[f(ˆR)] −
Pd−1
r=0 f(r)eµrad
Pd−1
r=0 eµrad
= 0,
(C.1)
where ad ≜√2 log d.
The proof of Proposition C.1 is based on an auxiliary result, which we prove in Sec-
tion C.1. To state this result, we introduce some additional notation. Let S(r), for r ∈
{0, 1, . . . , d −1}, be a discrete stochastic process. We define the function h(α)(r) as follows,
h(α)(r) ≜S(r) + αf(r),
(C.2)
where f(r) is a bounded deterministic function, and α ∈R. We further define,
Md(α) ≜max
r
h(α)(r),
(C.3)
and
ˆR(α) ≜arg max
r
h(α)(r).
(C.4)
Note that Md(a) and ˆR(a) are random variables. Finally, we denote ˆR ≜ˆR(0). We have the
following result, which is proved in Appendix C.1.
50

## Page 51

Lemma C.2. The following holds,
E[f(ˆR)] = d
dαE[Md(α)]

α=0
.
(C.5)
Lemma C.2 implies that finding the expected value of f(ˆR) is related directly to the
derivative of the expected value of the maximum around zero. Thus, the problem of finding
the expected value of f(ˆR) is related to finding the expected value of the maximum of the
stochastic process. In our case, S will be a Gaussian vector with mean given by (A.19) and
a covariance matrix given by (A.20). Thus, our goal now is to find the expected value of the
maximum of S. For this purpose, we will recall some well-known results on the maximum of
Gaussian processes.
It is known that for an i.i.d. sequence of normally distributed random variables {ξn}, the
asymptotic distribution of the maximum Mn ≜max{ξ1, ξ2, ..., ξn} is the Gumbel distribution,
i.e., for any x ∈R,
P [an(Mn −bn) ≤x] →e−e(−x),
(C.6)
as n →∞, where,
an ≜
p
2 log n
(C.7)
and,
bn ≜
p
2 log n −1
2
log log n + log 4π
√2 log n
.
(C.8)
It turns out that the above convergence result remains valid even if the sequence {ξn} is
not independent and normally distributed. Specifically, as shown in [29, Theorem 6.2.1], a
similar result holds for Gaussian random variables {ξn} with a covariance matrix that decays
such that limn→∞ρn ·log n = 0, and with a mean vector whose maximum value decays faster
than limn→∞max0≤m≤n−1 |µm|·√log n = 0. These conditions precisely match those specified
in Theorem 4.3.
C.1
Proof of Lemma C.2
The proof technique of Lemma C.2 is similar to the technique used in [39, 31], but with
a non-trivial adaption to the discrete case. To prove this lemma, we will first establish a
deterministic counterpart of (C.5). Specifically, we define,
h(α)(r) ≜X(r) + αf(r),
(C.9)
where r ∈{0, 1, . . . , d −1}. The functions X : {0, 1, . . . , d −1} →R, and f : {0, 1, . . . , d −
1} →R are assumed bounded and deterministic. We further assume that X is injective, i.e.,
for zi ̸= zj, we have X(zi) ̸= X(zj). Define,
s(α) ≜max
r {h(α)(r)},
(C.10)
51

## Page 52

and note that s(α) is well-defined over the supports of X and f, and it is a continuous
function of α around α = 0. Finally, we let,
Z(α)
max ≜arg max
r
{h(α)(r)}.
(C.11)
We have the following result.
Lemma C.3. The following relation holds,
d
dαs(α)

α=0
= f(Z(0)
max).
(C.12)
Proof of Lemma C.3. Note that,
d
dαs(α)

α=0
= lim
α→0
s(α) −s(0)
α
= lim
α→0
maxr[X(r) + αf(r)] −maxr X(r)
α
.
(C.13)
By the definition of Z(α)
max, we have,
max
r [X(r) + αf(r)] = X(Z(α)
max) + αf(Z(α)
max),
(C.14)
and
max
r
X(r) = X(Z(0)
max).
(C.15)
Now, the main observation here is that for a sufficiently small value of α around zero, we
must have that Z(α)
max and Z(0)
max equal because Z(0)
max can take discrete values only, and it is
unique. Thus, for α · maxr |f(r)| < mini̸=j |X(zi) −X(zj)|, we have,
Z(α)
max = Z(0)
max.
(C.16)
Combining (C.13)–(C.16) yields,
d
dαs(α)

α=0
= lim
α→0
X(Z(α)
max) + αf(Z(α)
max) −X(Z(0)
max)
α
(C.17)
= f(Z(0)
max),
(C.18)
which concludes the proof.
We are now in a position to prove Lemma C.2. Similarly to the deterministic case, we
define the random function,
h(α)(r) = S(r) + αf(r),
(C.19)
where S : {0, 1, . . . , d −1} →R is a discrete stochastic process, and f is a deterministic
function. We assume that S has a continuous probability distribution without any single
52

## Page 53

point with a measure greater than 0. Using Lemma C.3, for each realization of S(r), such
that S(r) is injective, we have,
f(ˆR) = d
dα Md(α)

α=0
.
(C.20)
Under the assumption above of S(r), the measure of the set of events that S is not injective
is zero. Therefore, the fact that Md(α)−Md(0)
α
is bounded (see, (C.34)) and (C.20), imply that,
E[f(ˆR)] =
Z
f(ˆR)dµ
=
Z
d
dα Md(α)

α=0
dµ
=
Z
lim
α→0
Md(α) −Md(0)
α

dµ
= lim
α→0
Z Md(α) −Md(0)
α

dµ
= d
dα EMd(α)

α=0
,
(C.21)
which concludes the proof.
C.2
Proof of Proposition C.1
Conditioned on N[k], the Gaussian vector S (see, (A.19) and (A.20)) can be represented as,
S|N[k] = Z + µ,
(C.22)
where Z is a zero mean Gaussian random vector with covariance matrix given by (A.20) and
µ is given by (A.19). Define,
h(α)(r) ≜Z(r) + µ(r) + αf(r),
(C.23)
where we use the same notations as in Lemma C.2. Then, using Lemma C.2,
E[f(ˆR)] = d
dαEMd(α)

α=0
,
(C.24)
where Md(α) = maxr {Z(r) + µ(r) + αf(r)}. Therefore, our goal is now to find the derivative
of EMd(α).
Using [29, Theorem 6.2.1], under the assumptions of Proposition C.1, for a sufficiently
small value of α such that limd→∞|α| maxr |f(r)| · √log d = 0, we have for any x ≥0,
lim
d→∞P [ad(Md(α) −bd −m⋆
d(α)) ≤x] = e−e(−x),
(C.25)
53

## Page 54

where ad and bd are given in (C.7) and (C.8), respectively, and
m⋆
d(α) ≜a−1
d log
 
d−1
d−1
X
i=0
ead(µi+αf(i))
!
.
(C.26)
For brevity, we denote,
Td(α) ≜ad · [Md(α) −bd −m⋆
d(α)],
(C.27)
and we note that,
Td(α) −Td(0) = ad[(Md(α) −m⋆
d(α)) −(Md(0) −m⋆
d(0))],
(C.28)
and so,
∆d(α) ≜1
ad
Td(α) −Td(0)
α
= Md(α) −Md(0)
α
−m⋆
d(α) −m⋆
d(0)
α
,
(C.29)
for any α ̸= 0. The following result shows ∆d(α) converges zero in the L1 sense.
Lemma C.4. For any α ̸= 0,
lim
d→∞|∆d(α)| = 0,
(C.30)
i.e., ∆d(α)
L1
−→0, as d →∞.
Proof of Lemma C.4. To prove (C.30), we will first show that ∆d(α) converges to zero in
probability.
Because ∆d(α) is uniformly integrable, this is sufficient for the desired L1
convergence above. Specifically, recall from (C.25) that Td(α) converges in distribution to
the Gumbel random variable Gum with location zero and unit scale, i.e., Td(α)
D−→Gum, as
d →∞. Furthermore, it is clear that
1
ad =
1
√2 log d →0, as d →∞. Thus, Slutsky’s theorem
[49] implies that,
Td(α)
ad
D−→0.
(C.31)
It is known that convergence in distribution to a constant implies also convergence in prob-
ability to the same constant [20], and thus,
Td(α)
ad
P−→0.
(C.32)
Therefore, the above result together with the continuous mapping theorem [20] implies that,
∆d(α)
P−→0,
(C.33)
for every α ̸= 0.
54

## Page 55

Next, we show that ∆d(α) is bounded with probability one. Indeed, by the definition of
Md(α) in (C.3), we have,

Md(α) −Md(0)
α
 ≤
max
0≤r≤d−1 |f(r)| < C < ∞,
(C.34)
for some C > 0, where we have used the fact that f is bounded. Furthermore, note that,
d
dαm⋆
d(α) =
Pd−1
i=0 f(i) exp{ad(µi + αf(i))
Pd−1
i=0 exp{ad(µi + αf(i))}
,
(C.35)
which is bounded because,

Pd−1
i=0 f(i) exp{ad(µi + αf(i))
Pd−1
i=0 exp{ad(µi + αf(i))}
 ≤
max
0≤r≤d−1 |f(r)| < C < ∞.
(C.36)
Combining (C.29), (C.34) and (C.36), leads to,
|∆d(α)| ≤

Md(α) −Md(0)
α
 +

m⋆
d(α) −m⋆
d(0)
α
 ≤2 max
0≤r≤d−1 |f(r)| < ∞.
(C.37)
Now, since ∆d(α) is bounded, it is also uniformly integrable, and thus when combined with
(C.33) we may conclude that,
∆d(α)
L1
−→0,
(C.38)
as claimed.
We continue with the proof of Proposition C.1. First, we show that,
lim
d→∞lim
α→0 E [∆d(α)] = lim
α→0 lim
d→∞E [∆d(α)] .
(C.39)
Indeed, note that,
lim
d→∞lim
α→0 E [∆d(α)] = lim
d→∞lim
α→0
Z Td(α) −Td(0)
α

dµ,
(C.40)
where dµ is the probability measure associated with Td. From (C.37) we know that ∆d(α)
is bounded. Thus, applying the dominated convergence theorem, we obtain,
lim
d→∞lim
α→0
Z Td(α) −Td(0)
α

dµ = lim
d→∞
Z
lim
α→0
Td(α) −Td(0)
α

dµ.
(C.41)
Since the integral at the right-hand-side of (C.41) is finite and bounded for each value of α,
and for each value of d, the order of the limits can be exchanged, thus leading to (C.39).
Therefore, from (C.29) and (C.39), we have,
lim
α→0 lim
d→∞E [∆d(α)] = lim
d→∞lim
α→0
E[Md(α) −Md(0)]
α
−[m⋆
d(α) −m⋆
d(0)]
α

(C.42)
55

## Page 56

= lim
d→∞
 d
dαEMd(α) −d
dαm⋆
d(α)

α=0
.
(C.43)
Now, Lemma C.4 implies that the left-hand-side of (C.42) nullifies, and thus,
lim
d→∞
 d
dαEMd(α) −d
dαm⋆
d(α)

α=0
= 0.
(C.44)
Finally, combining (C.35) and (C.44), we obtain (C.1), which concludes the proof.
D
Proof of Theorem 4.3
Remark on notation. In this section, we omit the dependence on 0 ≤i ≤M −1, where
this is clear from the context, e.g., Ni = N and ˆRi = ˆR. In addition, to streamline notation
we often omit the explicit dependence on the signal length and write x (and the associated
quantities) in place of x(d) whenever the dimension d is understood from the context.
D.1
Notations and auxiliary results
First, we introduce notation and present several auxiliary results that support Theorem 4.3.
Recall the definition of eXk[ℓ] from (A.21). We define:
σ2
k = σ2
2 ·
d−1
X
ℓ=0
|eXk[ℓ]|2,
(D.1)
which corresponds to the diagonal entries of Σk[r, s] in (A.20). In addition, recall the vector
S as defined in (A.5), and introduce the normalized vector:
˜Sk = S/σk,
(D.2)
where σk is given in (D.1). Then, by Lemma A.1, we have:
eSk|N[k] ∼N(eµk, eΣk),
(D.3)
where the mean and covariance are given by:
eµk[r] ≜2σ−1
k |X[k]| |Ni[k]| cos
2πkr
d
+ ϕNi[k] −ϕX[k]

(D.4)
for 0 ≤r ≤d −1, and
eΣk[r, s] ≜
Pd−1
ℓ=0 |eXk[ℓ]|2 cos
  2πℓ
d (r −s)

Pd−1
ℓ=0 |eXk[ℓ]|2
,
(D.5)
for 0 ≤r, s ≤d −1. Note in particular that normalizing S by σk ensures that the diagonal
entries of eΣk are equal to one.
Recall that we assume the template vector is normalized, i.e., Pd−1
ℓ=0 |X[ℓ]|2 = 1. Under
this assumption, we have the following lemma.
56

## Page 57

Lemma D.1. Suppose the conditions of Assumption 4.2 hold. Then, for all k ∈N, the
following limits hold:
lim
d→∞
d−1
X
ℓ=0
2 |X[ℓ]|2 −
eXk[ℓ]

2 = 0,
(D.6)
and,
lim
d→∞σ2
k = σ2.
(D.7)
Proof of Lemma D.1. From the definition of eXk[ℓ] in (A.21), we have:
2 |X[ℓ]|2 −
eXk[ℓ]

2
=





2 |X[ℓ]|2
ifℓ= k, d −k,
|X[ℓ]|2
ifℓ= 0, d/2,
0
otherwise.
(D.8)
According to Assumption 4.2, we have,
lim
d→∞

max
0<k≤d−1 {|X[k]|} ·
p
log d

= 0,
(D.9)
and X[0] = 0, Recall that the template vector is normalized, i.e., Pd−1
ℓ=0 |X[ℓ]|2 = 1, which
implies |X[ℓ]|2 ≤|X[ℓ]| for all 0 ≤ℓ≤d −1.
Combining (D.8)–(D.9), results,
lim
d→∞
d−1
X
ℓ=0

2 |X[ℓ]|2 −
eXk[ℓ]

2
= lim
d→∞

|X[0]|2 + |X[d/2]|2 + 4 |X[k]|2
= 0,
(D.10)
which proves (D.6).
To establish (D.7), observe that
lim
d→∞σ2
k = σ2
2 · lim
d→∞
d−1
X
ℓ=0
|eXk[ℓ]|2 = σ2,
(D.11)
where the final equality follows from (D.6). This completes the proof of the lemma.
We now state a lemma showing that the entries of the covariance matrix eΣk[r, s] satisfy
the conditions of Proposition C.1.
Lemma D.2. Suppose the conditions of Assumption 4.2 hold. Define,
ρ|r−s| ≜|eΣk [r, s] |,
(D.12)
for eΣk [r, s] defined in (D.5). Then, ρ0 = 1, and
ρn log (n) →0.
(D.13)
That is, the covariance matrix eΣk[r, s] satisfies the conditions required by Proposition C.1.
57

## Page 58

Proof of Lemma D.2. From the definition of the covariance matrix of eSk|N[k] in (D.5), we
observe that it is circulant and fully characterized by its eigenvalues |eXk[ℓ]|2 (see (A.21)) for
0 ≤ℓ≤d −1. Due to the normalization by σk, the covariance matrix is normalized such
that its diagonal entries equal one, i.e., ρ0 = 1.
It remains to show that the off-diagonal elements decay sufficiently fast, namely, ρm log (m) →
0, for m →∞.
Using the definition of ρ from (D.12), we can write:
ρm = eΣk [r, r −m] =
Pd−1
ℓ=0 |eXk[ℓ]|2 cos
  2πℓ
d m

Pd−1
ℓ=0 |eXk[ℓ]|2
.
(D.14)
As d →∞, the denominator in (D.14) converges to 2 by Lemma D.1.
The numerator
corresponds to the DFT of the sequence |eXk[ℓ]|2,
d−1
X
ℓ=0
|eXk[ℓ]|2 cos
2πℓ
d m

= F
n
|eXk|2o
[m].
(D.15)
By Lemma D.1, for each fixed k and for all m ∈{1, . . . , d −1},
F
 |eXk|2
[m] = 2 F
 |X|2
[m] + ∆k,d[m],
(D.16)
where the error term ∆k,d[m] is supported only on the modified frequencies ℓ∈{0, d/2, k, d−
k} and satisfies the uniform bound
sup
1≤m≤d−1
|∆k,d[m]| ≤|X[0]|2 + |X[d/2]|2 + 4|X[k]|2 −−−→
d→∞0,
(D.17)
using Assumption 4.2(2) and |X[0]| = 0 from Assumption 4.2(3).
Moreover, recalling that RXX = F
 |X|2
, Assumption 4.2(1) gives
max
1≤m≤d−1 log(d)
F
 |X|2
[m]
 =
max
1≤m≤d−1 log(d) |RXX[m]| −−−→
d→∞0.
(D.18)
Combining (D.16)–(D.17) with (D.18) yields
max
1≤m≤d−1 log(d)
F
 |eXk|2
[m]
 −−−→
d→∞0,
(D.19)
and together with (D.15) this implies the desired decay condition for the off-diagonal corre-
lations ρm, completing the proof.
D.2
High-dimensional limits
We now present a central result, which plays a key role in the proof of Theorem 4.3. To
state the result, we first define the functions:
f1(r) ≜|N[k]| cos
2πk
d r + ϕN[k] −ϕX[k]

,
(D.20)
58

## Page 59

and
f2(r) ≜|N[k]|2 sin2
2πk
d r + ϕN[k] −ϕX[k]

,
(D.21)
for 0 ≤r ≤d−1. Note that f1 and f2 correspond to the terms appearing in the expectation
in the denominator and numerator of (B.64), respectively.
Proposition D.3 (High-dimensional limits for f1(ˆR) and f2(ˆR) ). Assume the template
signal x satisfies Assumption 4.2, and that its DFT coefficients are non-vanishing, i.e.,
X[k] ̸= 0, for all 0 < k ≤d −1. Let f1(ˆR) and f2(ˆR) be defined as in (D.20) and (D.21),
with ˆR defined in (2.3). Then, as d →∞, their expected values satisfy:
lim
d→∞
1
ad · |X[k]|E[f1(ˆR)] = σ,
(D.22)
and
lim
d→∞E[f2(ˆR)] = σ2
2 ,
(D.23)
where ad ≜√2 log d.
The proof of Proposition D.3 builds on Proposition C.1 and the auxiliary Lemmas D.1–
D.2.
Proof of Proposition D.3. Our goal is to prove (D.22) and (D.23). By the law of total ex-
pectation, we have,
1
ad
E[f1(ˆR)] = 1
ad
E
h
E
h
f1(ˆR)
 N[k]
ii
,
(D.24)
and
E[f2(ˆR)] = E
h
E
h
f2(ˆR)
 N[k]
ii
.
(D.25)
Accordingly, we will prove
σk
ad · |X[k]|E
h
f1(ˆR)
 N[k]
i
L1
−→|N[k]|2 ,
(D.26)
and,
E
h
f2(ˆR)
 N[k]
i
L1
−→1
2 |N[k]|2 ,
(D.27)
which would yield the desired result.
To proceed, we apply Proposition C.1. Recall the definition of the vector ˜Sk given in
(D.2). Conditioned on N[k], this vector follows a Gaussian distribution with mean eµk[r] (as
defined in (D.4)) and covariance matrix eΣk[r, s] (defined in (D.5)). We assume that both
the mean and the covariance satisfy Assumption 4.2, and we assert that they also meet the
criteria of Proposition C.1. Indeed, observe the following:
59

## Page 60

1. The mean term. By Assumption 4.2, we have |X[k]| √log d →0, as d →∞, for
every 0 ≤k ≤d −1, implying that √log d max |eµk[r]| →0, where the |N[k]| term in
eµk[r] is finite and independent of d.
2. The covariance term. By Lemma D.2, the covariance matrix eΣk[r, s] satisfies the
conditions of Proposition C.1.
We apply Proposition C.1 and the result in (C.1) to the functions f1(ˆR) and f2(ˆR), with
respect to the Gaussian vector eSk|N[k] (D.3). Observe that
max
0≤r≤d−1 |eµk[r]| = 2σ−1
k |X[k]| |N[k]| ,
(D.28)
and,
f1(ˆR) = |N[k]| cos
2πk
d
ˆR + ϕN[k] −ϕX[k]

=
σk
2 |X[k]|eµk[ˆR].
(D.29)
Given that the assumptions of Proposition C.1 are satisfied, it follows that
E
h
f1(ˆR)
 N[k]
i
−
Pd−1
r=0 f1(r)eeµk[r]ad
Pd−1
r=0 eeµk[r]ad
a.s.
−−→0,
(D.30)
and,
E
h
f2(ˆR)
 N[k]
i
−
Pd−1
r=0 f2(r)eeµk[r]ad
Pd−1
r=0 eeµk[r]ad
a.s.
−−→0.
(D.31)
Next, we evaluate the terms at the left-hand-side of (D.30) and (D.31).
Proof of (D.26) and (D.22).
We begin by proving that
1
2ad
Pd−1
r=0 eµk[r] exp{eµk[r]ad}
Pd−1
r=0 exp{eµk[r]ad}
σ2
k
|X[k]|2
a.s.
−−→|N[k]|2 .
(D.32)
From the definition of f1(r), it follows that
d−1
X
r=0
f1(r) = 0,
(D.33)
almost surely. Additionally, from the definition of eµk[r],
eµk[r]ad = 2σ−1
k ad |X[k]| |N[k]| cos
2πk
d r + ϕN[k] −ϕX[k]

.
(D.34)
By Assumption 4.2, we have ad |X[k]| →0 as d →∞. Thus, from (D.34) and the continuous
mapping theorem, we obtain
eµk[r]ad
a.s.
−−→0.
(D.35)
60

## Page 61

Since Pd−1
r=0 eµk[r] = 0 almost surely (from (D.33)), and applying the continuous mapping
theorem along with (D.35), we deduce that
Pd−1
r=0 eµk[r] exp{eµk[r]ad}
ad
Pd−1
r=0[eµk[r]]2
a.s.
−−→1.
(D.36)
Similarly, applying the continuous mapping theorem using (D.35), we get
Pd−1
r=0 exp{eµk[r]ad}
d
a.s.
−−→1.
(D.37)
Next, from the definition of eµk[r],
d−1
X
r=0
[eµk[r]]2 = 4σ−2
k (|X[k]| |N[k]|)2
d−1
X
r=0
cos2
2πk
d r + ϕN[k] −ϕX[k]

.
(D.38)
As d →∞,
1
d
d−1
X
r=0
cos2
2πk
d r + ϕN[k] −ϕX[k]

a.s.
−−→1
2.
(D.39)
Combining (D.38) and (D.39), we obtain
σ2
k
2 |X[k]|2
d−1
X
r=0
[eµk[r]]2
a.s.
−−→|N[k]|2 ,
(D.40)
for d →∞. Thus, combining (D.36)–(D.40), we conclude that
σ2
k
2 |X[k]|2 ad
Pd−1
r=0 eµk[r]eeµk[r]ad
Pd−1
r=0 eeµk[r]ad
a.s.
−−→|N[k]|2 .
(D.41)
This proves (D.32). Finally, combining (D.29), (D.30), and (D.41), we arrive at
σ2
k
2 |X[k]|2 ad
E
h
eµk[ˆR]
 N[k]
i
a.s.
−−→|N[k]|2 .
(D.42)
Let the term on the left-hand side of (D.42) be denoted by Gd and the term on the
right-hand side by G, so that Gd
a.s.
−−→G. By definition, observe that |Gd| ≤|N[k]|2, and
it is evident that E[|N[k]|2] = σ2 < ∞. Thus, by the dominated convergence theorem, we
conclude that Gd
L1
−→G, and specifically,
E
"
σ2
k
2ad
E[µ(ˆR)|N[k]]
|X[k]|2
#
→E

|N[k]|2
.
(D.43)
By combining (D.43) with (D.29), we get,
σk
ad · |X[k]|E
h
f1(ˆR)
 N[k]
i
L1
−→|N[k]|2 ,
(D.44)
61

## Page 62

or equivalently,
E

σk
ad · |X[k]|E
h
f1(ˆR)
 N[k]
i
→E

|N[k]|2
,
(D.45)
which proves (D.26).
By the law of total expectation, we then obtain
σk
ad · |X[k]|E

|N[k]| cos
2πk
d
ˆR + ϕN[k] −ϕX[k]

→E

|N[k]|2
= σ2,
(D.46)
as d →∞. By Lemma D.1, we know that σ2
k →σ2 as d →∞, so from (D.46), we conclude
that
1
ad · |X[k]|E

|N[k]| cos
2πk
d
ˆR + ϕN[k] −ϕX[k]

→σ,
(D.47)
which proves (D.22).
Proof of (D.27) and (D.23).
The numerator in (B.64) converges to,
E[|N[k]| sin(ϕe[k])]2 = 1
2

E |N[k]|2 −E[|N[k]|2 cos(2ϕe[k])]

→1
2E |N[k]|2 ,
(D.48)
as d →∞, where the last transition is because E[cos(2ϕe[k])|N[k]]
a.s.
−−→0, as d →∞. Thus,
1
σ2E[|N[k]| sin(ϕe[k])]2 →
1
2σ2E |N[k]|2 = 1
2,
(D.49)
which concludes the proof.
D.3
Proof of Theorem 4.3
To begin, let us summarize the notation and results from the previous sections as the foun-
dation for the proof. Recall the definition of ϕe[k] in (A.7), as well as the definitions of
f1 (r) and f2 (r) in (D.20)-(D.21), and let ad ≜√2 log d. According to Theorem 4.1, the
convergence of the Fourier phases is given by:
lim
M→∞
E|ϕˆX[k] −ϕX[k]|2
1/M
= Ck.
(D.50)
where Ck is given by (B.64):
Ck = E

[|N[k]| sin(ϕe[k])]2
(E[|N[k]| cos(ϕe[k])])2 .
(D.51)
The constant Ck can be rewritten as:
a2
d · Ck = a2
d · E

[|N[k]| sin(ϕe[k])]2
(E[|N[k]| cos(ϕe[k])])2 = a2
d ·
E[f2(ˆR)]

E[f1(ˆR)]
2,
(D.52)
62

## Page 63

where f1 (r) and f2 (r) are defined in (D.20)-(D.21).
The signal x, satisfying the conditions of Theorem 4.3, also satisfies the assumptions of
Proposition D.3. By Proposition D.3, we have:
lim
d→∞
1
ad · |X[k]|E[f1(ˆR)] = σ,
(D.53)
and,
lim
d→∞E[f2(ˆR)] = σ2
2 .
(D.54)
Now, we are ready to prove the results of the Theorem.
Convergence of the Fourier phases.
From (D.52)–(D.54), we obtain:
lim
d→∞a2
d · |X[k]|2 · Ck = 1
2.
(D.55)
Combining (D.50), (D.55), results,
lim
d→∞lim
M→∞
E|ϕˆX[k] −ϕX[k]|2
1/M
1
1/(4 log (d) |X[k]|2)
= lim
d→∞
Ck
1/(4 log (d) |X[k]|2)
(D.56)
= lim
d→∞
a2
d · |X[k]|2 · Ck
1/2
= 1,
(D.57)
where (D.56) follows from (D.50), and (D.57) follows from (D.55), proving (4.8).
Convergence of the Fourier magnitudes.
Finally, we prove (4.9). By Theorem 4.1 and
(4.3), we have:
|ˆX[k]|
a.s.
−−→E

|N[k]| cos
2πk
d
ˆR + ϕN[k] −ϕX[k]

= E[f1(ˆR)].
(D.58)
Combining (D.53), (D.58) yields,
1
adσ
|ˆX[k]|
|X[k]|
a.s.
−−→
1
adσ
E[f1(ˆR)]
|X[k]|
→1,
(D.59)
as M, d →∞, where the second passage follows from (D.22). As ad =
p
2 log (d), this
completes the proof of the Theorem.
E
Proof of Proposition 5.1
Before proving Proposition 5.1, we first establish the following auxiliary lemma.
Lemma E.1. Let A = (A0, A1, . . . , Ad−1) be a d-dimensional random vector with E[A] = 0.
Then,
E [max{A0, A1, . . . , Ad−1}] ≥
max
0≤r1,r2≤d−1
1
2E [|Ar1 −Ar2|] .
(E.1)
63

## Page 64

Proof of Lemma E.1. For any two real numbers x and y, we have:
max(x, y) = 1
2(x + y + |x −y|).
(E.2)
Applying this to any pair Ar1, Ar2 yields:
E[max{Ar1, Ar2}] = 1
2E[Ar1 + Ar2 + |Ar1 −Ar2|]
(E.3)
= 1
2E[|Ar1 −Ar2|],
(E.4)
where we used the assumption that E[Ar] = 0 for all r.
By the convexity of the max function, it holds that:
E[max{A0, A1, . . . , Ad−1}] ≥E[max{Ar1, Ar2}],
(E.5)
for every r1, r2 ∈{0, 1, . . . , d −1}. Combining (E.4) and (E.5), we conclude:
E[max{A0, A1, . . . , Ad−1}] ≥
max
0≤r1,r2≤d−1
1
2E[|Ar1 −Ar2|],
(E.6)
completing the proof.
Let n0, n1, . . . , nM−1 be an i.i.d. sequence of zero-mean random vectors with covariance
E[nin⊤
i ] = Σ, which by assumption Σ is positive-definite. Recall the definition of the EfN
estimator in (2.4):
ˆx ≜1
M
M−1
X
i=0
T−ˆRini,
(E.7)
where the estimated shift ˆRi is given by:
ˆRi ≜arg max
0≤ℓ≤d−1
⟨ni, Tℓx⟩.
(E.8)
Using linearity of the inner product:
⟨ˆx, x⟩=
*
1
M
M−1
X
i=0
T−ˆRini, x
+
= 1
M
M−1
X
i=0
⟨T−ˆRini, x⟩.
(E.9)
By SLLN, as M →∞, we have almost surely:
1
M
M−1
X
i=0
⟨T−ˆRini, x⟩
a.s.
−−→E[⟨T−ˆR1n1, x⟩].
(E.10)
Define for r ∈{0, 1, . . . , d −1} the random variables:
Ar ≜⟨n1, Trx⟩.
(E.11)
64

## Page 65

Then, the right-hand side of (E.10) becomes:
E[⟨T−ˆR1n1, x⟩] = E[max{A0, A1, . . . , Ad−1}].
(E.12)
Applying Lemma E.1, we get:
E[max{A0, A1, . . . , Ad−1}] ≥
max
0≤r1,r2≤d−1
1
2E[|Ar1 −Ar2|]
(E.13)
=
max
0≤r1,r2≤d−1
1
2E[|⟨n1, Tr1x −Tr2x⟩|].
(E.14)
To complete the proof, we show that the lower bound in (E.14) is strictly positive. Since
x ∈Rd is nonzero with non-vanishing Fourier components X[k] ̸= 0, for every 1 ≤k ≤d −1,
and Tr is a cyclic shift operator, the set {Trx : 0 ≤r < d} contains at least d −1 distinct
vectors. Thus, there exist r1, r2 ∈{0, . . . , d −1} such that
v ≜Tr1x −Tr2x ̸= 0.
(E.15)
Then the inner product ⟨n1, v⟩is a real-valued random variable with
Var(⟨n1, v⟩) = E[⟨n1, v⟩2] = v⊤Σv > 0,
(E.16)
because v ̸= 0 and Σ is positive definite. Hence, ⟨n1, v⟩is not almost surely zero, and
E[|⟨n1, v⟩|] > 0.
(E.17)
This implies that
max
0≤r1,r2<d
1
2E[|⟨n1, Tr1x −Tr2x⟩|] > 0,
(E.18)
and consequently,
lim
M→∞⟨ˆx, x⟩= E[⟨T−ˆR1n1, x⟩] > 0,
(E.19)
almost surely. This completes the proof.
F
Proof of Theorem 5.2: High-dimensional i.i.d. noise
In this section, we prove Theorem 5.2.
The proof relies on the functional central limit
theorem for the discrete Fourier transform [37, 14, 13], which we review in Appendix F.1.
In Appendices F.2 and F.3, we apply this result to analyze the real and imaginary parts of
the EfN estimator under a general i.i.d. noise model, and compare the outcome to the white
Gaussian case. Finally, the proof of Theorem 5.2 is deduced in Appendix F.4.
65

## Page 66

F.1
The functional CLT for DFT
We begin by presenting a functional central limit theorem (CLT) for the DFT, which estab-
lishes that the DFT of an i.i.d. real-valued sequence converges in distribution as the dimen-
sion d →∞. This result has been studied in the literature; see, for example, [37, 14, 13]. To
formalize this, we state the following functional CLT for DFTs of i.i.d. sequences.
Theorem F.1 (Functional CLT for the DFT). Let {zn}n∈N be a sequence of i.i.d. real-valued
random variables with zero mean E[z0] = 0 and finite variance E[z2
0] = σ2 < ∞. For each
integer d ≥1, define the DFT of the finite segment {z0, . . . , zd−1} as
Z(d)[k] ≜
1
√
d
d−1
X
ℓ=0
zℓe−2πjkℓ/d,
0 ≤k < d.
(F.1)
Extend Z(d) to an infinite sequence by zero-padding outside the index set {0, . . . , d −1}:
Z(d) =
 Z(d)[0], . . . , Z(d)[d −1], 0, 0, . . .

∈CN.
(F.2)
Then, for any fixed finite index set {k1, k2, . . . , km} ⊂N, the finite-dimensional vectors
 Z(d)[k1], . . . , Z(d)[km]

converge in distribution, as d →∞,
 Z(d)[k1], . . . , Z(d)[km]

D
−−−→
d→∞
 Wk1, . . . , Wkm

,
(F.3)
where W = (Wk)k∈N is a sequence of i.i.d. circularly symmetric complex Gaussian random
variables with Wk ∼CN(0, σ2).
This result can be obtained from the multivariate Lindeberg–Feller CLT. This conver-
gence holds jointly over any finite collection of indices, meaning that for every finite subset
I ⊂N, the finite-dimensional vector (Z(d)
k )k∈I converges in distribution to (Wk)k∈I, where
the Wk are i.i.d. circularly symmetric complex Gaussian random variables. The collection of
these finite-dimensional distributions is consistent and satisfies the compatibility conditions
of Kolmogorov’s extension theorem, thereby uniquely determining a probability law on the
infinite product space CN. In this sense, the convergence of Z(d) to W is fully characterized
by convergence of finite-dimensional distributions.
F.2
Notations
General i.i.d noise.
Let z0, z1, . . . be a sequence of i.i.d random variables with E[z0] = 0,
and E[z2
0] = σ2 < ∞, and E [z4
0] < ∞. Define the (zero-padded) DFT transform of the finite
segment z(d) = (z0, z1, . . . , zd−1) as
Z(d)[k] ≜





1
√
d
d−1
X
ℓ=0
zℓe−2πjkℓ/d,
0 ≤k < d,
0,
otherwise.
(F.4)
66

## Page 67

Let x = (x0, . . . , xd−1) be the deterministic template signal, and denote its DFT by X[k].
Define the maximal correlation shift between x and z(d) in the Fourier domain as
ˆR(d)
Z
≜arg max
0≤r<d
d−1
X
k=0
|X[k]| |Z[k]| cos
2πkr
d
+ ϕZ[k] −ϕX[k]

,
(F.5)
where ϕZ[k] and ϕX[k] are the phases of Z(d)[k] and X[k], respectively. Define the phase
difference after alignment as
ϕ(d)
e,Z[k] ≜2πkˆR(d)
Z
d
+ ϕZ[k] −ϕX[k].
(F.6)
Gaussian i.i.d noise.
Let n0, n1, . . . be an i.i.d sequence of Gaussian random variables
with nℓ∼N(0, σ2). Define the DFT of the segment n(d) = (n0, . . . , nd−1) as
N(d)[k] ≜





1
√
d
d−1
X
ℓ=0
nℓe−2πjkℓ/d,
0 ≤k < d,
0,
otherwise.
(F.7)
Define the corresponding maximal correlation shift:
ˆR(d)
N ≜arg max
0≤r<d
d−1
X
k=0
|X[k]| |N[k]| cos
2πkr
d
+ ϕN[k] −ϕX[k]

,
(F.8)
and define the aligned phase difference as
ϕ(d)
e,N[k] ≜2πkˆR(d)
N
d
+ ϕN[k] −ϕX[k].
(F.9)
F.3
Convergence of the real and imaginary parts of the EfN esti-
mator
We now present an auxiliary result that relates the real and imaginary parts of the EfN
estimator under both Gaussian i.i.d.
and general i.i.d.
noise models.
This result is a
consequence of the functional central limit theorem for the DFT (Theorem F.1).
Proposition F.2. Let z0, z1, . . . be a sequence of i.i.d. real-valued random variables with
zero mean, finite variance, E[z2
0] = σ2 < ∞, and finite fourth moment. Let n0, n1, . . . be an
i.i.d. sequence of Gaussian random variables with nℓ∼N(0, σ2). Let Z(d) and N(d) denote
the DFTs of the sequences {zi}d−1
i=0 and {ni}d−1
i=0 , respectively. Let ϕ(d)
e,Z[k] and ϕ(d)
e,N[k] denote
the aligned phase differences defined in (F.6) and (F.9), respectively. Then, for each fixed
frequency index k ∈N,
lim
d→∞

E
hZ(d)[k]
 sin

ϕ(d)
e,Z[k]
i
−E
hN(d)[k]
 sin

ϕ(d)
e,N[k]
i
= 0,
(F.10)
lim
d→∞

E
hZ(d)[k]
 cos

ϕ(d)
e,Z[k]
i
−E
hN(d)[k]
 cos

ϕ(d)
e,N[k]
i
= 0,
(F.11)
67

## Page 68

and
lim
d→∞

Var
hZ(d)[k]
 sin

ϕ(d)
e,Z[k]
i
−Var
hN(d)[k]
 sin

ϕ(d)
e,N[k]
i
= 0,
(F.12)
lim
d→∞

Var
hZ(d)[k]
 cos

ϕ(d)
e,Z[k]
i
−Var
hN(d)[k]
 cos

ϕ(d)
e,N[k]
i
= 0.
(F.13)
Proof of Proposition F.2. We begin by applying Theorem F.1 to the sequence {zi}, which
satisfies its assumptions. This in turn implies that,
Z(d)
D−→W,
(F.14)
where Wk ∼CN(0, σ2) are i.i.d. complex Gaussian variables. Next, consider the Gaussian
noise sequence ni ∼N(0, σ2). Its discrete Fourier transform satisfies
N(d) D= (W[0], . . . , W[d −1]),
(F.15)
for every d, since the DFT of an i.i.d. Gaussian sequence remains i.i.d. in distribution with
the same complex Gaussian law. Since E[z2
i ] = σ2 < ∞, we have
E[|Z(d)[k]|2] < ∞,
(F.16)
for each d and k. It follows that the sequences of random variables,
n
|Z(d)[k]| sin(ϕ(d)
e,Z[k])
o
d∈N+ ,
n
|Z(d)[k]| cos(ϕ(d)
e,Z[k])
o
d∈N+ ,
(F.17)
are uniformly integrable. By Vitali’s convergence theorem, we may pass the limit inside the
expectation,
lim
d→∞

E
h
|Z(d)[k]| sin(ϕ(d)
e,Z[k])
i
−E [|W[k]| sin(ϕe,W[k])]

= 0.
(F.18)
Furthermore, since N(d) D= W for all d, and the aligned phase differences ϕ(d)
e,N[k] and ϕe,W[k]
are identically distributed, we conclude that
E
h
|N(d)[k]| sin(ϕ(d)
e,N[k])
i
= E [|W[k]| sin(ϕe,W[k])] ,
∀d ∈N+.
(F.19)
Combining (F.18) and (F.19) yields
lim
d→∞

E
h
|Z(d)[k]| sin(ϕ(d)
e,Z[k])
i
−E
h
|N(d)[k]| sin(ϕ(d)
e,N[k])
i
= 0,
(F.20)
which establishes (F.10). An identical argument with sine replaced by cosine proves (F.11).
For the variance convergence, by assumption E[z4
i ] < ∞, and so,
E[|Z(d)[k]|4] < ∞,
(F.21)
68

## Page 69

for all d. Therefore, the sequences
n
|Z(d)[k]|2 sin2(ϕ(d)
e,Z[k])
o
d∈N+ ,
n
|Z(d)[k]|2 cos2(ϕ(d)
e,Z[k])
o
d∈N+ ,
(F.22)
are uniformly integrable. Applying Vitali’s theorem once again, we obtain,
lim
d→∞

Var
h
|Z(d)[k]| sin(ϕ(d)
e,Z[k])
i
−Var [|W[k]| sin(ϕe,W[k])]

= 0.
(F.23)
As before, since N(d) D= W and their aligned phase differences are identically distributed, we
conclude that,
Var
h
|N(d)[k]| sin(ϕ(d)
e,N[k])
i
= Var [|W[k]| sin(ϕe,W[k])] ,
∀d ∈N+.
(F.24)
Combining (F.23) and (F.24) yields,
lim
d→∞

Var
h
|Z(d)[k]| sin(ϕ(d)
e,Z[k])
i
−Var
h
|N(d)[k]| sin(ϕ(d)
e,N[k])
i
= 0,
(F.25)
establishing (F.12). Repeating the same steps with cosine in place of sine proves (F.13).
F.4
Proof of Theorem 5.2
We are now ready to prove Theorem 5.2. As before, let {zi}M−1
i=0
be i.i.d. observations, where
each zi ∈Rd has i.i.d. entries with zero mean, finite variance, and bounded fourth moment
E[(zi[ℓ])4] < ∞, for all ℓ∈{0, 1, . . . , d −1}.
Similarly to (A.13), we analyze the EfN estimator under the noise statistics of {zi}M−1
i=0 .
Applying the SLLN, as M →∞, we obtain:
ˆX[k]e−jϕX[k] = 1
M
M−1
X
i=0
|Zi[k]| ejϕe,Zi[k]
(F.26)
a.s.
−−→E [|Z1[k]| cos (ϕe,Z1[k])] + jE [|Z1[k]| sin (ϕe,Z1[k])] ,
(F.27)
where the phase difference term is given by
ϕ(d)
e,Zi[k] ≜2πkˆR(d)
Zi
d
+ ϕZi[k] −ϕX[k].
(F.28)
and the corresponding maximal correlation shift ˆR(d)
Zi is defined by
ˆR(d)
Zi ≜arg max
0≤r<d
d−1
X
k=0
|X[k]| |Zi[k]| cos
2πkr
d
+ ϕZi[k] −ϕX[k]

.
(F.29)
Next, we invoke Proposition F.2, whose assumptions are satisfied in this setting. Let N(d)
1
denotes a noise vector with i.i.d. Gaussian entries that match the first and second moments
69

## Page 70

of the entries of Z(d)
1
(as defined in Proposition F.2). We note that the results of Theorems
4.1 and 4.3 apply to the case of i.i.d. Gaussian entries N(d)
1 . Then, by Proposition F.2, for
each fixed frequency index k ∈N, the following convergence results hold,
lim
d→∞

E
hZ(d)
1 [k]
 sin

ϕ(d)
e,Z1[k]
i
−E
hN(d)
1 [k]
 sin

ϕ(d)
e,N1[k]
i
= 0,
(F.30)
lim
d→∞

E
hZ(d)
1 [k]
 cos

ϕ(d)
e,Z1[k]
i
−E
hN(d)
1 [k]
 cos

ϕ(d)
e,N1[k]
i
= 0.
(F.31)
Moreover, the variances of the corresponding expressions also converge,
lim
d→∞

Var
hZ(d)
1 [k]
 sin

ϕ(d)
e,Z1[k]
i
−Var
hN(d)
1 [k]
 sin

ϕ(d)
e,N1[k]
i
= 0,
(F.32)
lim
d→∞

Var
hZ(d)
1 [k]
 cos

ϕ(d)
e,Z1[k]
i
−Var
hN(d)[k]
 cos

ϕ(d)
e,N1[k]
i
= 0.
(F.33)
By Theorems 4.1 and 4.3, the convergence behavior of the estimator is governed by the
variances in (F.32) and (F.33).
Therefore, the asymptotic behavior of the estimator for
general i.i.d. noise {zi} matches that of the Gaussian i.i.d. case {ni}. In particular for (5.3),
we have,
ϕˆX[k] −ϕX[k] = arctan


PM−1
i=0
Z(d)
i [k]
 sin (ϕe,Zi[k])
PM−1
i=0
Z(d)
i [k]
 cos (ϕe,Zi[k])


(F.34)
a.s.
−−→arctan


E
hZ(d)
1 [k]
 sin (ϕe,Z1[k])
i
E
hZ(d)
1 [k]
 cos (ϕe,Z1[k])
i

,
(F.35)
where (F.35) follows from the SLLN as M →∞. Applying (F.30) and (F.31) into (F.35),
yields,
lim
d→∞lim
M→∞ϕˆX[k] −ϕX[k] = lim
d→∞arctan


E
hN(d)
1 [k]
 sin (ϕe,N1[k])
i
E
hN(d)
1 [k]
 cos (ϕe,N1[k])
i

.
(F.36)
By (B.62), the r.h.s. of (F.36) vanishes for every d, and therefore,
lim
d→∞lim
M→∞ϕˆX[k] −ϕX[k] = 0,
(F.37)
almost surely, which proves (5.3). Similarly, for (5.4), we have,
lim
M→∞
E|ϕˆX[k] −ϕX[k]|2
1/M
= E
 [|Z1[k]| sin(ϕe,Z1[k])]2
(E[|Z1[k]| cos(ϕe,Z1[k])])2 .
(F.38)
70

## Page 71

which is similar to (B.24). Applying (F.31) and (F.32) into (F.38) yields,
lim
d→∞lim
M→∞
E|ϕˆX[k] −ϕX[k]|2
1/M
= lim
d→∞
E
 [|N1[k]| sin(ϕe,N1[k])]2
(E[|N1[k]| cos(ϕe,N1[k])])2 = Ck < ∞.
(F.39)
Finally, for (5.5), under the assumption that x satisfies Assumption 4.2, then by (4.8), the
r.h.s. of (F.39) converges to,
lim
d→∞a2
d · |X[k]|2 · Ck = 1
2,
(F.40)
where ad =
p
2 log(d). Thus, substituting (F.40) into the r.h.s. of (F.39) yields,
lim
d→∞lim
M→∞
E [|ϕˆX[k] −ϕX[k]|2]
1/(M log d)
·
1
1/(4|X[k]|2) = 1,
(F.41)
which proves (5.5).
G
Proof of Proposition 5.4: Circulant Gaussian noise
The proof strategy for Proposition 5.4 closely follows that of the i.i.d. Gaussian case (The-
orem 4.1), with appropriate modifications to handle circulant noise.
The necessary as-
sumptions and notations are introduced in Appendix G.1. Appendix G.2 establishes the
asymptotic convergence of the EfN estimator as M →∞under circulant Gaussian noise
statistics. In Appendix G.3, we show that conditioning the EfN process on a single Fourier
noise coefficient results in a cyclo-stationary process with a cosine trend. Appendix G.4
extends the vanishing imaginary part result from Appendix B.1 to the setting of circulant
Gaussian noise. Similarly, Appendix G.5 extends the result of Appendix B.2, showing that
the real part remains strictly positive in the circulant case. Finally, Appendix G.6 combines
the results of the preceding sections to complete the proof of Proposition 5.4.
G.1
Preliminaries
Let {yi}M−1
i=0
∼N(0, Σ), where Σ ∈Rd×d is a real, symmetric, and circulant covariance
matrix with strictly positive eigenvalues (i.e., Σ is positive-definite). Let Yi = F {yi} ∈Cd
denote the DFT of yi. The random vector Yi satisfies the following properties:
1. Diagonalization by the DFT. Since Σ is circulant, it is diagonalized by the DFT: Σ =
F ∗ΛF, where F is the DFT matrix, Λ = diag(λ0, . . . , λd−1) contains the eigenvalues
of Σ, given by the DFT of its first row. As Σ positive-definite, all eigenvalues λk ∈R
and λk > 0 for all k ∈{0, 1, . . . d −1}.
2. Distribution of Fourier coefficients. The vector Yi is complex Gaussian with distribu-
tion CN(0, Λ). Its entries are independent (but not identically distributed) complex
Gaussian random variables, satisfying, E[Yi[k]] = 0, and E[Yi[k]Yi[ℓ]] = λkδk,ℓ, for
every k, ℓ∈{0, 1, . . . d/2}.
71

## Page 72

3. Fourier phases distribution. For any k such that λk > 0, the Fourier coefficient Yi[k]
is a zero-mean, circularly symmetric complex Gaussian random variable. Hence, the
phases {ϕYi[k]}d/2
k=0 are i.i.d. and uniformly distributed on [−π, π) and independent of
the magnitude {|Yi[k]|}d/2
k=0.
4. Conjugate symmetry. Since yi ∈Rd, the DFT satisfies the Hermitian symmetry:
Yi[d −k] = Yi[k],
for 1 ≤k ≤d −1.
(G.1)
Thus, only the first d/2+1 Fourier coefficients are independent; the rest are determined
by conjugate symmetry.
Remark G.1. To avoid confusion with the i.i.d. Gaussian case, we use the notation yi and
Yi, rather than ni and Ni, to denote Gaussian noise with a symmetric circulant covariance
matrix.
G.2
The convergence of the Einstein from Noise estimator
Similar to the derivation in Appendix A.2, the EfN estimator in the setting of circulant
Gaussian noise, {yi}M−1
i=0
∼N (0, Σ), can be expressed explicitly as:
ˆX[k] = 1
M
M−1
X
i=0
|Yi[k]| ejϕYi[k] ej 2πk
d ˆRi
(G.2)
= ejϕX[k]
M
M−1
X
i=0
|Yi[k]| ejϕYi[k] ej 2πk
d ˆRi e−jϕX[k]
(G.3)
= ejϕX[k]
M
M−1
X
i=0
|Yi[k]| ejϕe,i[k],
(G.4)
where the shifts ˆRi are given by
ˆRi ≜arg max
0≤r≤d−1
⟨yi, Trx⟩
(G.5)
= arg max
0≤r≤d−1
⟨F {yi} , F {Trx}⟩
(G.6)
= arg max
0≤r≤d−1
d−1
X
k=0
|X[k]| |Yi[k]| cos
2πkr
d
+ ϕYi[k] −ϕX[k]

.
(G.7)
and the phase difference is defined as,
ϕe,i[k] ≜2πkˆRi
d
+ ϕYi[k] −ϕX[k].
(G.8)
To simplify notation, define for each r ∈{0, 1, . . . , d −1},
Si[r] ≜
d−1
X
k=0
|X[k]| |Yi[k]| cos
2πkr
d
+ ϕYi[k] −ϕX[k]

,
(G.9)
72

## Page 73

so that ˆRi = arg max0≤r≤d−1 Si[r]. We note that for any 0 ≤i ≤M −1, the random vector
Si ≜(Si[0], Si[1], . . . , Si[d−1])⊤is jointly Gaussian with zero mean and a circulant covariance
matrix (as it is a Fourier transform of the convolution between yi and the template x). Hence,
Si forms a cyclo-stationary process. Applying the strong law of large numbers (SLLN), as
M →∞, we have,
ˆX[k] e−jϕX[k] = 1
M
M−1
X
i=0
|Yi[k]| ejϕe,i[k]
(G.10)
a.s.
−−→E [|Y1[k]| cos (ϕe,1[k])] + j E [|Y1[k]| sin (ϕe,1[k])] ,
(G.11)
where we have used the fact that the sequences of random variables {|Yi[k]| cos(ϕe,i[k])}M−1
i=0
and {|Yi[k]| sin(ϕe,i[k])}M−1
i=0
are i.i.d. with finite means and variances. Finally, we define for
each k,
µA,k ≜E [|Y1[k]| sin (ϕe,1[k])] ,
(G.12)
µB,k ≜E [|Y1[k]| cos (ϕe,1[k])] ,
(G.13)
as the asymptotic imaginary and real parts of ˆX[k]e−jϕX[k], respectively.
G.3
Conditioning on the Fourier frequency noise component
We now extend the result of Lemma A.1 to the case where the noise follows a general
Gaussian distribution with a real, symmetric, and circulant covariance matrix. That is, we
consider observations {yi}M−1
i=0
∼N(0, Σ), where Σ ∈Rd×d is circulant and symmetric. In
this setting, we establish the following result.
Lemma G.2. Let Si be defined as in (G.9), and denote E [|Yi[k]|2] = λk > 0 for each
k ∈{0, 1, . . . , d −1}. Then, for every k ∈

1, 2, . . . , d
2 −1, d
2 + 1, . . . , d −1
	
, the random
vector Si conditioned on Yi[k] is Gaussian:
Si|Yi[k] ∼N(µk,i, Σk,i),
(G.14)
with mean and covariance given by,
µk,i[r] ≜E [Si[r]|Yi[k]] = 2 |X[k]| |Yi[k]| cos
2πkr
d
+ ϕYi[k] −ϕX[k]

,
(G.15)
for 0 ≤r ≤d −1, and
Σk,i[r, s] ≜E [(Si[r] −ESi[r]) (Si[s] −ESi[s]) |Yi[k]]
=
d−1
X
ℓ=0
λℓ
2 · |eXk[ℓ]|2 cos
2πℓ
d (r −s)

,
(G.16)
for 0 ≤r, s ≤d −1, where eXk is defined by:
eXk[ℓ] ≜





0
ifℓ= k, d −k,
X[ℓ]
ifℓ= 0, d/2,
√
2 · X[ℓ]
otherwise.
(G.17)
73

## Page 74

Note that the conditional process Si|Yi[k] is Gaussian because it is given by a linear
transform of i.i.d. Gaussian variables. Also, since its covariance matrix is circulant and
depends only on the difference between the two indices, i.e., Σk,i[r, s] = σk,i[|r −s|], it is
cycle-stationary with a cosine trend. The eigenvalues of this circulant matrix are given by
the DFT of its first row, and thus its ℓ-th eigenvalue equals λℓ· |eXk[ℓ]|2, for 0 ≤ℓ≤d −1.
Remark G.3. When the noise is i.i.d. Gaussian, that is, yi ∼N(0, σ2Id×d), the eigenvalues
of the covariance matrix satisfy λℓ= σ2 for all ℓ∈{0, 1, . . . , d−1}. In this case, the general
setting reduces to the one considered in Lemma A.1, thereby recovering its result.
Proof of Lemma G.2. We recall that if {yi}M−1
i=0
∼N(0, Σ), for symmetric circulant matrix
Σ, then their DFT coefficients satisfy {|Yi[k]|}d/2
k=0, and {ϕYi[k]}d/2
k=0 are independent and
{ϕYi[k]}d−1
k=0 ∼Unif[−π, π). By definition of Si (G.9), we have for every k ̸= 0, d/2,
Si [r] |Yi[k] =2 |X[k]| |Yi[k]| cos
2πkr
d
+ ϕYi[k] −ϕX[k]

+
X
ℓ̸=k,d−k
|X[ℓ]| |Yi[ℓ]| cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

,
(G.18)
where we have used the property of X[k] = X[d −k], Yi[k] = Yi[d −k]. Clearly, as E [Yi [ℓ]] =
0, for every 0 ≤ℓ≤d −1, we have,
E

|X[ℓ]| |Yi[ℓ]| cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

= 0,
(G.19)
for every 0 ≤ℓ≤d −1. Combining (G.18) and (G.19) results,
µk,i[r] = E [Si[r]|Yi[k]] = 2 |X[k]| |Yi[k]| cos
2πkr
d
+ ϕYi[k] −ϕX[k]

,
(G.20)
proving the first result concerning the means.
The covariance term.
In the following, we derive the covariance term,
Σk,i[r, s] ≜E [(Si[r] −ESi[r]) (Si[s] −ESi[s]) |Yi[k]] .
(G.21)
Let,
ρk,i [r] ≜Si[r] −ESi[r]
=
X
ℓ̸=k,d−k
|X[ℓ]| |Yi[ℓ]| cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

,
(G.22)
and denote,
I = {1, 2, . . . k −1, k + 1, . . . , d/2 −1} ,
(G.23)
74

## Page 75

which is the indices of the Fourier coefficients, excluding {0, k, d/2}.
As the sequences
{|Yi[ℓ]|}d/2
ℓ=0 and {ϕYi[ℓ]}d/2
ℓ=0 are statistically independent, and satisfy Yi[ℓ] = Yi[d −ℓ] and
X[ℓ] = X[d −ℓ], we have,
ρk,i [r] =
X
ℓ̸=k,d−k
|X[ℓ]| |Yi[ℓ]| cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

=
=
X
ℓ∈{0,d/2}
|X[ℓ]| |Yi[ℓ]| cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

+ 2 ·
X
ℓ∈I
|X[ℓ]| |Yi[ℓ]| cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

,
(G.24)
where each one of the terms in the sum is independent. Since the terms in the sum on the
r.h.s. of (G.24) are independent (i.e., E
h
Yi [ℓ1] Yi [ℓ2]
i
= E

|Yi [ℓ1]|2
δℓ1,ℓ2), it follows that,
Σk,i[r, s] = E [ρk,i [r] ρk,i [s] |Yi[k]]
= E


X
ℓ∈{0,d/2}
|X[ℓ]|2 |Yi[ℓ]|2 cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

cos
2πℓs
d
+ ϕYi[ℓ] −ϕX[ℓ]


+ 4 · E
"X
ℓ∈I
|X[ℓ]|2 |Yi[ℓ]|2 cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

cos
2πℓs
d
+ ϕYi[ℓ] −ϕX[ℓ]
#
.
(G.25)
The expectation value in (G.25) is composed of the multiplications of cosines. Applying
trigonometric identities, we obtain,
cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

cos
2πℓs
d
+ ϕYi[ℓ] −ϕX[ℓ]

= 1
2 cos
2πℓ(r −s)
d

+ 1
2 cos
2πℓ(r + s)
d
+ 2 (ϕYi[ℓ] −ϕX[ℓ])

,
(G.26)
for every 0 ≤r, s ≤d −1. Now, since the sequences {|Yi[ℓ]|}d/2
ℓ=0 and {ϕYi[ℓ]}d/2
ℓ=0 are indepen-
dent random variables, with E

|Yi[k]|2
= λk and phases ϕYi[k] uniformly distributed over
[−π, π), and by applying the trigonometric identity (G.26), it follows that,
E

|Yi[ℓ]|2 cos
2πℓr
d
+ ϕYi[ℓ] −ϕX[ℓ]

cos
2πℓs
d
+ ϕYi[ℓ] −ϕX[ℓ]

= 1
2E

|Yi[ℓ]|2
cos
2πℓ(r −s)
d

= λℓ
2 cos
2πℓ(r −s)
d

.
(G.27)
Substituting (G.27) into (G.25) leads to,
E [ρk,i [r] ρk,i [s] |Yi[k]] =
X
ℓ∈{0,d/2}
λℓ
2 · |X[ℓ]|2 cos
2πℓ
d (r −s)

75

## Page 76

+ 4 ·
X
ℓ∈I
λℓ
2 · |X[ℓ]|2 cos
2πℓ
d (r −s)

.
(G.28)
As for every ℓ∈I, |X[ℓ]| = |X[d −ℓ]|, we have,
4 ·
X
ℓ∈I
λℓ
2 |X[ℓ]|2 cos
2πℓ
d (r −s)

= 2
X
ℓ̸={0,k,d/2,d−k}
λℓ
2 |X[ℓ]|2 cos
2πℓ
d (r −s)

.
(G.29)
Substituting (G.29) into (G.28), we get,
E [ρk,i [r] ρk,i [s] |Yi[k]] =
d−1
X
ℓ=0
λℓ
2 · |eXk[ℓ]|2 cos
2πℓ
d (r −s)

,
(G.30)
for eXk[ℓ] as defined in (G.17), which completes the proof.
G.4
Convergence of the Fourier phases
Similarly to Appendix B.1 and Lemma B.1, we show here that the imaginary part in (G.11)
vanishes. The key observation is that {|Yi[k]|}d−1
k=0, and {ϕYi[k]}d−1
k=0 are statistically indepen-
dent and {ϕYi[k]}d−1
k=0 ∼Unif[−π, π).
Lemma G.4. Recall the definition of ϕe,i[k] in (G.8). Then,
µA,k = E [|Y1[k]| sin(ϕe,1[k])] = 0,
(G.31)
for every 0 ≤k ≤d −1.
Proof of Lemma G.4. Let D[k] ≜ϕX[k] −ϕY1[k], and recall the definition of ˆRi in (A.4), i.e.,
ˆRi = arg max
0≤r≤d−1
d−1
X
k=0
|X[k]| |Yi[k]| cos
2πkr
d
+ ϕYi[k] −ϕX[k]

.
(G.32)
Note that ˆRi is a function of
ˆRi = ˆRi

{|Yi[k]|}d−1
k=0 , {|X[k]|}d−1
k=0 , {ϕYi[k]}d−1
k=0 , {ϕX[k]}d−1
k=0

,
(G.33)
and it depends on ϕYi[k] and ϕX[k] only through D[k]. Accordingly, viewing ˆR1 as a function
of D[k], for fixed {|Yi[k]|}d−1
k=0 , {|X[k]|}d−1
k=0, we have,
ˆR1 (−D[0], −D[1], . . . , −D[d −1]) = −ˆR1 (D[0], D[1], . . . , D[d −1]) .
(G.34)
Namely, from symmetry arguments, by flipping the signs of all the phases, the location of
the maximum flips its sign as well. Then, by the law of total expectation,
µA,k = E

|Y1[k]| sin
2πk
d
ˆR1 + ϕY1[k] −ϕX[k]

76

## Page 77

= E

|Y1[k]| · E

sin
2πk
d
ˆR1 + ϕY1[k] −ϕX[k]
 {|Y1[k]|}d−1
k=0

.
(G.35)
The inner expectation in (G.35) is taken w.r.t.
the uniform randomness of the phases
{ϕY1[k]}d−1
k=0 ∈[−π, π). However, due to (G.34), and since the sine function is odd around
zero, the integration in (G.35) nullifies. Therefore,
E

sin
2πk
d
ˆR1 + ϕY1[k] −ϕX[k]
 {|Y1[k]|}d−1
k=0

= 0,
(G.36)
and thus µA,k = 0.
G.5
Convergence to non-vanishing signal
In analogy with Appendix B.2 and Proposition B.2, we now establish that the real part
of (G.11) does not vanish.
Proposition G.5. Recall the definition of ϕe,i[k] in (G.8). Fix d ∈N, and assume that
X[k] ̸= 0 for all 0 < k ≤d −1. Then, for any 0 ≤k ≤d −1,
µB,k ≜E[|Y1[k]| cos(ϕe,1[k])] > 0.
(G.37)
Proof of Proposition G.5. By the law of total expectation, we have,
E[|Y1[k]| cos(ϕe,1[k])] = E [|Y1[k]| · E (cos(ϕe,1[k])| Y1[k])]
= E
"
|Y1[k]| · E
 
cos
 
2πkˆR1
d
+ ϕY1[k] −ϕX[k]
! Y1[k]
!#
.
(G.38)
More explicitly, we can write,
E[|Y1[k]| cos(ϕe,1[k])] =
1
2π
Z ∞
0
dy yf|Y1[k]|(y)
Z π
−π
dφE

cos
2πk
d
ˆR1 + φ
 |Y1[k]| = y, ϕY1[k] = ϕX[k] + φ

.
(G.39)
Now, note that the inner integral can be written as,
Z π
−π
dφE

cos
2πk
d
ˆR1 + φ
 |Y1[k]| = y, ϕY1[k] = ϕX[k] + φ

=
Z π
0
dφ E

cos
2πk
d
ˆR1 + φ
 |Y1[k]| = y, ϕY1[k] = ϕX[k] + φ

+
+
Z π
0
dφ E

cos
2πk
d
ˆR1 + φ + π
 |Y1[k]| = y, ϕY1[k] = ϕX[k] + φ + π

.
(G.40)
Next, we apply Proposition A.5. Using its notation, we define the Gaussian process,
S(+) = S1|Y1[k],
(G.41)
77

## Page 78

where the r.h.s. follows from (G.14). By (G.15), the mean vector of S1|Y1[k] has a cosine
trend, as assumed in Proposition A.5 in (A.59). Additionally, S1|Y1[k] is a Gaussian cyclo-
stationary process, as described in (G.16). The final condition to verify is (A.60), which is
satisfied by Lemma A.4 (and applies also to circulant Gaussian noise statistics as well).
Since the conditional distribution of ˆR1 given {|Y1[k]| = y, ϕY1[k] = ϕX[k] + φ} matches
that of ˆR(+) in (A.57), and similarly, given {|Y1[k]| = y, ϕY1[k] = ϕX[k] + φ + π}, it matches
ˆR(−) in (A.58), the sum of the integrands on the right-hand side of (G.40) equals the left-
hand side of (A.63). By Proposition A.5, this sum is positive for all φ ∈[0, π]. Together
with (G.39), this completes the proof of Proposition G.5.
G.6
Proof of Proposition 5.4
We are now ready to prove Proposition 5.4. By the definition of the phase difference between
the template x and the EfN estimator ˆx (as in (A.6)), we have,
ϕˆX[k] −ϕX[k] = arctan
 PM−1
i=0 |Yi[k]| sin (ϕe,i[k])
PM−1
i=0 |Yi[k]| cos (ϕe,i[k])
!
,
(G.42)
Using the continuous mapping theorem, it is evident that it suffices to prove that,
PM−1
i=0 |Yi[k]| sin (ϕe,i[k])
PM−1
i=0 |Yi[k]| cos (ϕe,i[k])
a.s.
−−→0.
(G.43)
This, however, follows by applying the SLLN,
PM−1
i=0 |Yi[k]| sin (ϕe,i[k])
PM−1
i=0 |Yi[k]| cos (ϕe,i[k])
a.s.
−−→µA,k
µB,k
,
(G.44)
where µA,k ≜E [|Y1[k]| sin(ϕe,1[k])] and µB,k ≜E [|Y1[k]| cos(ϕe,1[k])], defined in (G.12), and
(G.13), respectively. By Lemma G.2, µA,k = 0, while by Proposition G.5, we have that
µB,k > 0, and thus their ratio converges a.s. to zero by the continuous mapping theorem.
Thus, we proved that ϕˆX[k]
a.s.
−−→ϕX[k]. Finally, we prove the convergence rate, given in (5.5).
According to Proposition B.4, whose assumptions apply for the case of circulant Gaussian
noise statistics as well, we have,
lim
M→∞
E|ϕˆX[k] −ϕX[k]|2
1/M
= E
 [|Y1[k]| sin(ϕe,1[k])]2
E[|Y1[k]| cos(ϕe,1[k])]2
< ∞,
(G.45)
which completes the proof of the Proposition.
78
