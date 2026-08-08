# SpecRaGE-LIU v1 — registered development specification

**Status:** implementation specification; experiments are forbidden until the
independent code/method review is complete.

**Canonical name:** **SpecRaGE-weighted Laplacian-regularized IU-PCR**.

**Method source:** Amitai Yacobi, Ofir Lindenbaum, and Uri Shaham,
*Generalizable and Robust Spectral Method for Multi-view Representation
Learning*, TMLR 2025, <https://arxiv.org/abs/2411.02138>.

## 1. Scientific question

The current DUFS-LIU graph assigns one global feature metric to every sample.
It improves IU-PCR when that geometry is target-aligned, but it follows a
smooth nuisance manifold just as readily.  SpecRaGE-LIU asks a narrower
question:

> When feature-family reliability changes from sample to sample, can
> self-supervised spectral agreement learn which family to trust locally and
> thereby construct a safer Laplacian for IU-PCR than DUFS does?

The intended contribution is conditional graph reliability.  This method does
not claim to identify a nuisance shared coherently by every feature family.

## 2. Data-access contract

The per-cell learner accepts only:

- the fixed `fixed_stable_v1`, confidence-oriented feature matrix;
- frozen feature names and provenance view assignments;
- numerical configuration and random seeds.

It does not accept correctness labels, AUROC, target latents, a target-derived
graph, or a target-dependent stopping decision.  Labels are joined only after
all scores for a registered configuration are frozen.

For this development cycle, labels may select one **global** configuration
using grouped, cross-cell calibration.  Consequently the full research
procedure is described as *unsupervised per-cell learning with supervised
development-time hyperparameter calibration*, not as a strictly unsupervised
end-to-end procedure.  A later baseline can replace this calibration with a
strictly self-supervised selection rule.

Random sample splits within a cell are not independent validation units and may
not be used to select scientific hyperparameters.  Dataset/model families are
the selection units.  New families remain untouched confirmation data.

## 3. Frozen provenance views

View assignment follows primitive source, never feature AUROC:

1. entropy level;
2. entropy-trajectory dynamics;
3. sampled-token/spilled-energy trajectory;
4. log-partition energy trajectory;
5. top-k distribution summaries;
6. structural information such as trace length.

The exact mapping is versioned in `spectral_utils/specrage_views.py`.  Moving a
feature between views is a new method version, not a tuning operation.

## 4. Mathematical definition

For each view (v), an encoder produces

\[
Y^{(v)}=g_v(X^{(v)}).
\]

A fusion network maps the concatenated views to sample-specific simplex
weights

\[
\alpha_i=\operatorname{softmax}(q(X_i)/\tau),
\quad \alpha_i^{(v)}\ge 0,
\quad \sum_v\alpha_i^{(v)}=1,
\]

and produces

\[
y_i=\sum_v \alpha_i^{(v)}y_i^{(v)}.
\]

The first implementation uses the paper's Gaussian k-nearest-neighbour
affinity on each standardized, low-dimensional view:

\[
W_{ij}^{(v)}=\exp\left(
 -\frac{\lVert x_i^{(v)}-x_j^{(v)}\rVert^2}{2\sigma_v^2}
\right)
\]

on registered neighbour edges, with \(\sigma_v\) equal to the global median
neighbour distance.  Siamese affinity learning is deliberately deferred until
the sample-specific fusion mechanism is established.

SpecRaGE reliability modifies every view affinity as

\[
\widetilde W_{ij}^{(v)}=
W_{ij}^{(v)}\alpha_i^{(v)}\alpha_j^{(v)}.
\]

The view encoders use the released implementation's final `tanh` activation;
the fusion logits remain linear before the temperature softmax. The optimizer
uses no weight decay. The encoders and fusion network minimize the paper's
weighted Rayleigh loss under the released implementation's batch-normalized QR
layer,

\[
Y^\top Y/m=I,
\]

implemented by the detached transform \(\sqrt m R^{-1}\). Orthogonalization and
gradient updates use distinct unlabeled batches. Learning-rate scheduling,
early stopping, and checkpoint selection use a fixed unlabeled validation
subset and never correctness labels. The graph passed to IU-PCR is

\[
W_{\mathrm{SR}}=\frac{1}{V}\sum_v\widetilde W^{(v)}.
\]

Let (L_{\mathrm{SR}}) be its symmetric normalized Laplacian.  For the
ordinary IU-PCR feature matrix (F\), covariance (C=FF^\top/n\), target
moment estimate \(\hat\rho\), and fixed leading two-dimensional covariance
subspace (U\), define

\[
R_{\mathrm{SR}}=\frac1nFL_{\mathrm{SR}}F^\top
\]

and trace-match this roughness to (C\) inside (U\).  The only modification to
IU-PCR is

\[
w_\lambda=
U\left[U^\top(C+\lambda\bar R_{\mathrm{SR}})U\right]^{-1}
U^\top\hat\rho.
\]

At \(\lambda=0\), scores and weights must reproduce ordinary IU-PCR bitwise.

## 5. Primary assumptions and measurements

| assumption | diagnostic |
|---|---|
| provenance groups behave as distinct views | view membership and cross-view neighbourhood overlap |
| reliability varies by sample | distribution and entropy of \(\alpha_i\); sample-specific versus global control |
| at least one view remains useful when another is corrupted | planted corruption-to-weight recovery in synthetic worlds |
| useful geometry is shared more consistently than view-specific nuisance | view-specific corruption and nuisance-world performance |
| view Laplacians possess an approximate common basis | Laplacian commutators and off-diagonal energy of \(Y^\top L^{(v)}Y\) |
| learned weights are reproducible | seed-to-seed alpha deviation, graph stability, and score stability |
| the graph changes IU-PCR through structured roughness | projected roughness spectrum, score energy, weight angle, and rank displacement |

Collapse to one view, uniformly high entropy, seed instability, disconnected
graphs, or a large train-loss decrease without stable weights are failures to
be reported, not implementation details to conceal.

## 6. Registered controls

Headline methods:

1. current deployed U-PCR on the identical feature contract;
2. frozen DUFS-LIU (`k=7`, `lambda=0.1`), independent of every SpecRaGE
   candidate setting;
3. SpecRaGE-LIU.

Mechanism controls:

1. ordinary IU-PCR (exact \(\lambda=0\) anchor);
2. uniform view weights;
3. one learned global weight per view;
4. learned sample-specific weights;
5. sample-permuted learned weights;
6. trace-matched projected ridge;
7. oracle latent/corruption controls in synthetic data only.

The sample-specific arm must separate from the global and permuted arms before
the result can be attributed to conditional reliability.

## 7. Hyperparameter discipline

Only four scientific quantities may use labeled development calibration:

- neighbour count \(l\);
- spectral output dimension \(k\);
- softmax temperature \(\tau\);
- IU-PCR regularization \(\lambda\).

The search is a small registered set, not an unrestricted Cartesian sweep.
Learning rate, batch size, stopping, and architecture may be changed only to
repair documented convergence failures and are chosen without AUROC.

Configuration selection uses held-out dataset/model families. Means, standard
errors, and bootstrap intervals treat the dataset/model family as the
independent unit; cells within a family are averaged first. The chosen
configuration is the least complex configuration within one standard error of
the best cross-fitted equal-family-macro improvement over deployed U-PCR,
subject to:

- no orientation inversions;
- finite results and passed algebra invariants;
- no worse median result than DUFS-LIU;
- no graph collapse, defined before execution as effective-edge fraction
  `<0.10`, fifth-percentile degree / mean degree `<0.01`, or near-isolated
  sample fraction `>0.05`;
- reported lower-tail and worst-cell degradation.

If no candidate passes every guard, selection halts. There is no fallback to an
invalid configuration.

After selection, configuration, source hash, view schema, and seeds are frozen.

## 8. Stage gates

### Gate A — code and method review

Before any test or experiment is executed, an independent sub-agent must review:

- fidelity to SpecRaGE Eq. 4–5 and the QR procedure;
- correctness of weighted graph construction;
- exact IU-PCR integration;
- leakage boundaries and calibration logic;
- numerical and reproducibility risks;
- whether the controls isolate sample-specific reliability.

Every blocking finding must be fixed or explicitly accepted before execution.

### Gate B — known-answer and smoke

Required invariants:

- alpha rows are nonnegative and sum to one;
- graphs are finite, symmetric, and nonnegative;
- \(\lambda=0\) is exact IU-PCR;
- fixed-alpha graph construction is permutation-equivariant;
- same seed is deterministic;
- multi-seed headline and controls use the identical seed-mean-alpha graph
  operator;
- the real artifact exports `__labels` for every cell;
- labels cannot enter the learner API.

Only after these pass may a two-replicate synthetic smoke run execute.

### Gate C — synthetic mechanism

The candidate is evaluated on:

- clean aligned views;
- sample-specific view corruption;
- globally corrupted view;
- view-specific nuisance;
- shared unmeasured nuisance;
- pure noise.

The initial checkpoint reports convergence, alpha reliance, controls, and
performance plots.  Real calibration is paused until this checkpoint is
interpreted jointly.

### Gate D — grouped real calibration and confirmation

If Gate C supports the mechanism, current cells are development data for
grouped calibration.  A frozen configuration is then evaluated on newly
collected, untouched dataset/model families.  Labels enter only the evaluator.

### Gate E — independent result review

A different sub-agent reviews raw result tables and the registered gates,
checks leakage and alternative explanations, and writes its critique before the
project conclusion is updated.

## 9. Claim boundary

A positive result supports:

> Sample-specific, self-supervised spectral view weighting can construct a
> more useful IU-PCR regularizer when corruption is asymmetric across feature
> families.

It does not support target identification under an unmeasured nuisance shared
by all views.  Failure in that world is expected to remain visible in every
report and plot.
