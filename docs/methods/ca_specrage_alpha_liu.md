# CA-SpecRaGE-alpha-LIU

## Terms used below

`CA` means **cross-view agreement**. A view agrees with another view at a sample
when their short diffusion neighbourhoods are similar. A **reliability target**
is a label-free probability distribution over views. `Alpha` is SpecRaGE's
sample-specific view weight. The alpha graph connects two samples strongly only
when a view connects them and both samples rely on that view.

## Source and novelty boundary

The encoders, dynamic fusion weights, orthogonalized output, and spectral
Rayleigh loss are based on Yacobi, Lindenbaum, and Shaham,
[Generalizable and Robust Spectral Method for Multi-view Representation Learning](https://arxiv.org/abs/2411.02138).
The IU-PCR head is based on Tenzer et al.,
[Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html).

The agreement target, its KL loss, the edge-mass safeguard, the alpha-weighted
full-data graph, and their use inside LIU are project contributions. The exact
method below is not described or validated in either source.

## Label-free cross-view agreement

For each view (v), build a sparse affinity (W^{(v)}) and row-normalize it to
a transition matrix (P^{(v)}). Use a short diffusion profile

\[
H^{(v)}=\tfrac12P^{(v)}+\tfrac12(P^{(v)})^2. \tag{1}
\]

Each row of (H^{(v)}) is L2-normalized. For sample (i), view (v)'s mean
agreement with the other views is

\[
a_i^{(v)}=\sum_{u\ne v}\frac{q_u}{1-q_v}
\left\langle H_{i:}^{(v)},H_{i:}^{(u)}\right\rangle. \tag{2}
\]

For at least three views, the label-free target is

\[
\pi_i=\operatorname{softmax}
\left(a_i/\tau_a+\log q\right), \tag{3}
\]

where (q) is the registered prior mass of each view. Thus a duplicate-balanced
atomic view does not receive the same voting power as a complete micro-view.
With only two disagreeing views, agreement cannot identify which one is better.
The implementation returns (q) in that case.

For each seed, the target is built only on that seed's registered unlabeled fit
pool (at most 1,500 samples). Training and validation rows are disjoint subsets
of this pool. After fitting, the network predicts alpha for every sample. This
keeps target construction and neural fitting under the same sample budget.
Because target preprocessing sees the complete fit pool before that split, the
validation loss is an unsupervised optimization diagnostic. It is not an
inductively untouched validation or confirmation set.

## CA training objective

SpecRaGE predicts sample-specific weights

\[
\alpha_i=\operatorname{softmax}(q_\phi(x_i)/\tau). \tag{4}
\]

We add agreement and edge-mass terms to its spectral loss:

\[
\mathcal L=
\mathcal L_{\mathrm{SpecRaGE}}
+\beta\,\frac1n\sum_i\operatorname{KL}(\pi_i\Vert\alpha_i)
+\gamma\,\mathcal L_{\mathrm{mass}}. \tag{5}
\]

The mass term compares each alpha-weighted view's retained affinity mass with
the mass expected from its mean alpha. It prevents the optimizer from lowering
the Rayleigh loss by assigning adjacent endpoints to different views and
silently deleting most graph edges.

## Alpha graph and LIU

To compare schemas with different numbers of views, define relative reliability
(r_i^{(v)}=\alpha_i^{(v)}/q_v). The derived alpha graph is

\[
W_\alpha=\sum_{v=1}^V q_v
\operatorname{diag}(r^{(v)})W^{(v)}
\operatorname{diag}(r^{(v)}). \tag{6}
\]

If the learner returns its prior, Equation (6) is exactly the prior-weighted
average of the view graphs. This prevents a schema with 26 atomic views from
having a different baseline edge scale than one with six manual views.

Its normalized Laplacian (L_\alpha) defines

\[
R_\alpha=FL_\alpha F^\top/n. \tag{7}
\]

After trace matching in the IU-PCR subspace,

\[
w_\lambda=
U\left[U^\top(C+\lambda\bar R_\alpha)U\right]^{-1}
U^\top\hat\rho. \tag{8}
\]

At (lambda=0), it is exactly IU-PCR.

## What the method is trying to solve

IU-PCR assumes one reliability value per feature and pairwise uncorrelated
errors. Our data suggest a harder situation: a feature family can be useful for
some samples and irrelevant or misleading for others. CA-SpecRaGE tries to
estimate this local relevance without labels. It trusts a view when its local
neighbourhood is supported by other views, then discourages the final score from
varying rapidly across those trusted connections.

This addresses conditional relevance only if cross-view agreement is related to
correctness. Agreement can also describe a nuisance shared by several views.

## View construction experiment

The benchmark now compares:

1. **Manual:** the six old provenance families. This is the baseline.
2. **Atomic:** one feature per view. Features in the same learned micro-cluster
   divide one cluster's prior mass, so near-duplicates do not gain more total
   influence merely because several versions exist.
3. **Micro:** small feature groups learned without labels from their effect on
   the IU-PCR fusion subspace.

For feature (j), a one-feature graph gives (L_j) and

\[
R_j=FL_jF^\top/n,\qquad S_j=U^\top R_jU/\operatorname{tr}(U^\top R_jU). \tag{9}
\]

The raw (S_j) coordinates are not compared across cells because eigenvector
sign and basis can change. Inside each cell we compute

\[
d_{jk}=\|S_j-S_k\|_F/\sqrt{2}. \tag{10}
\]

This pairwise distance is unchanged by a common orthogonal change of basis.
For a held cell, distances are aggregated from the other 23 cells only. Average
linkage is applied for candidate cluster counts 3--8. A fixed label-free score
combines distance silhouette, sample/cell bootstrap adjusted-Rand stability,
singleton fraction, and cluster-size imbalance. All candidate scores and the
chosen partition are saved before labels are opened.

## Frozen hyperparameters

| parameter | value |
|---|---:|
| output dimension | 2 |
| base-graph neighbours | 15 |
| fusion temperature | 1 |
| agreement temperature (	au_a) | 0.08 |
| agreement coefficient (eta) | 2.0 |
| edge-mass coefficient (gamma) | 0.1 |
| encoder/fusion hidden width | 32 / 50 |
| learning rate | 0.01 |
| batch size / epochs | 128 / 60 |
| seeds | 11, 23 |
| unlabeled fit-sample cap | 1,500 |
| view schemas | manual, balanced atomic, LOCO micro |
| micro candidate groups | 3--8 |
| micro cluster bootstraps | 40 |
| impact-profile sample bootstraps | 4 at 80% |
| maximum samples per impact profile | 1,500 |
| view-mass normalization | on |
| orthogonalization | SVD floor (10^{-3}) |
| headline LIU lambda | 10 |

The earlier synthetic study selected (lambda=10). A ten-cell execution pilot
later suggested that smaller lambda may look better on real data. The frozen
headline remains 10 to measure honest transfer. Smaller values are shown only
as a clearly labeled sensitivity path.

## Checks required

- agreement-target entropy and variation by sample/view;
- micro-view silhouette, adjusted-Rand stability, singleton fraction, and
  partition transfer from the other 23 cells;
- alpha entropy, alpha-target distance, view switching, and seed stability;
- retained edge mass, effective edge fraction, components, degree tails, and
  spectral gap;
- comparison with an exact prior-alpha graph, global alpha, end-to-end uniform
  fusion, permuted alpha graph, and the graph on the CA-trained embedding (Y);
- projected roughness, condition number, weight cosine, score energy, and rank
  displacement relative to IU-PCR;
- convergence, gradient norms, and SVD-floor activation;
- AUROC and AUPRC by cell/domain/family, paired intervals, wins/losses, and the
  worst-cell change.

The method has passed a planted synthetic mechanism test. That does not prove
that real feature-view agreement follows correctness.

## Computational cost

Training has the same leading cost as adapted SpecRaGE. Agreement adds sparse
diffusion products: if every base graph has about (nk) edges, (P^2) can grow
toward (O(nk^2)) nonzeros per view. The code records graph density and collapse
diagnostics. The final alpha graph has about (O(Vnk)) candidate edges, and the
LIU solve has the same cost as in DUFS-LIU.
