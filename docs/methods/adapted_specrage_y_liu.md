# Adapted SpecRaGE-Y-LIU

## Terms used below

A **view** is a group of related features. A view-specific encoder maps its
features into a small representation. A fusion network produces weights
(alpha_i^{(v)}) telling how much sample (i) uses view (v). An
**embedding** (Y) gives every sample a learned low-dimensional coordinate.
LIU builds a graph from these coordinates and uses it to regularize IU-PCR.

## Paper basis and citation status

The representation learner is based on Amitai Yacobi, Ofir Lindenbaum, and Uri
Shaham, [Generalizable and Robust Spectral Method for Multi-view Representation Learning](https://arxiv.org/abs/2411.02138),
arXiv:2411.02138. The repository cites it as a manuscript. The benchmark is an
adaptation, not a reproduction of the paper's published experiments.

The graph-on-(Y) plus LIU fusion is our connection to Tenzer et al.,
[Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html).
It is not part of the SpecRaGE manuscript.

## SpecRaGE mathematics

For view (v), an encoder produces

\[
y_i^{(v)}=g_{\theta_v}(x_i^{(v)})\in\mathbb R^k. \tag{1}
\]

The fusion network produces a simplex weight for every sample,

\[
\alpha_i=\operatorname{softmax}(q_\phi(x_i)/\tau),
\qquad
\sum_v\alpha_i^{(v)}=1. \tag{2}
\]

The unorthogonalized fused representation is

\[
\widetilde y_i=\sum_v\alpha_i^{(v)}y_i^{(v)}. \tag{3}
\]

After batch orthogonalization, the output (Y) is trained with a multi-view
Rayleigh loss. In the manuscript's arithmetic-mean form,

\[
\mathcal L_{\mathrm{SpecRaGE}}
=\frac{2}{B^2V}\operatorname{tr}
\left(Y^\top\sum_{v=1}^V L^{(v)}Y\right). \tag{4}
\]

In the dynamic form, each view affinity is weighted at both endpoints by
(alpha_i^{(v)}\alpha_j^{(v)}). The objective learns a common spectral
representation without correctness labels.

## How it enters our fusion algorithm

The benchmark builds a (k)-nearest-neighbour graph (W_Y) on the final
two-dimensional embedding (Y). It then applies exactly the same LIU equation
as DUFS-LIU:

\[
w_\lambda=
U\left[U^\top(C+\lambda\bar R_Y)U\right]^{-1}
U^\top\hat\rho,
\qquad R_Y=FL_YF^\top/n. \tag{5}
\]

The embedding is used only to construct a sample graph. IU-PCR still supplies
(hat\rho), the PCR subspace, and the final score.

## Exact changes from the manuscript

- Views are fixed groups of hallucination features, not different data
  modalities supplied by a benchmark. The experiment compares manual,
  duplicate-balanced atomic, and leave-one-cell-out micro-view definitions.
- Gaussian self-tuning affinities replace the manuscript's learned Siamese
  affinity preprocessing.
- Output dimension is 2 because the LIU head is two-dimensional.
- An SVD singular-value floor replaces QR when the orthogonalization batch is
  ill-conditioned. The raw condition number and clipping are recorded.
- The network uses one hidden layer per encoder and fusion model, fixed CPU
  training, and two seeds.
- Training uses at most 1,500 unlabeled samples per cell and then forwards every
  sample. This tests the manuscript's out-of-sample mapping and bounds runtime.
- A registered view prior and view-mass normalization make total graph mass
  comparable when the number of views changes.
- A graph is built on (Y), then passed into a separate IU-PCR solver.

For these reasons the method is named **Adapted SpecRaGE-Y-LIU**, not simply
“SpecRaGE.” Its loss has no CA agreement or edge-mass terms.

## Assumptions and failure modes

1. The provenance groups are meaningful views of a shared sample geometry.
2. Their Laplacians have useful approximate joint low-frequency directions.
3. Dynamic fusion can downweight corrupted views using only the spectral loss.
4. Neighbours in (Y) should have similar correctness scores.
5. The learned geometry affects IU-PCR inside its two-PC subspace.

The spectral loss may identify a real but target-irrelevant nuisance geometry.
It may also leave (alpha) near uniform, create an unstable orthogonalization,
or learn a graph that changes no score ranks.

## Frozen hyperparameters

| parameter | value |
|---|---:|
| output dimension | 2 |
| graph neighbours | 15 |
| fusion temperature | 90 |
| encoder hidden width | 32 |
| fusion hidden width | 50 |
| learning rate | 0.01 |
| batch size | 128 |
| epochs | 60 |
| seeds | 11, 23 |
| unlabeled fit-sample cap | 1,500 |
| view-mass normalization | on |
| orthogonalization | SVD floor at (10^{-3}) of the leading singular value |
| agreement coefficient | 0 |
| edge-mass coefficient | 0 |
| headline LIU lambda | 10 |

The lambda value is the earlier synthetic transfer setting for the SpecRaGE
family. It is not selected from the 24-cell result.

## Checks required

- training loss and gradient convergence for both seeds;
- raw orthogonalization condition number and clipping fraction;
- weight entropy, dominant-view fraction, and seed disagreement;
- embedding-graph components, degree distribution, edge overlap, and seed
  stability;
- comparison with end-to-end uniform fusion;
- exact LIU identity at (lambda=0), projected condition number, and rank
  displacement;
- AUROC and AUPRC with paired domain/family and lower-tail reporting.

## Computational cost

The manuscript reports near-linear complexity in sample count when architecture,
epochs, views, and batch size are fixed. In this implementation, each training
batch also builds (V) pairwise affinities, giving an approximate cost
(O(S T (n/B) V B^2 d)) for (S) seeds and view dimension (d), plus neural
forward/backward work. Full-data sparse graph construction and neighbour search
are additional. Stored graph memory is about (O(nk)); neural activations are
about (O(VBk)) per batch.
