# DUFS-LIU: canonical baseline and design source

**Status (2026-08-07):** DUFS-LIU is the project's canonical **strictly
label-free Laplacian baseline** and a mathematical design source. It is no
longer described as the leading candidate. In the frozen 24-cell benchmark it
scored 0.7741 macro AUROC, only +0.008pp versus IU-PCR. Labels were read only
after every score was frozen.

## Canonical name

Use **DUFS-LIU** as shorthand for:

> **DUFS-gated Laplacian-regularized IU-PCR**

The components mean:

- **DUFS:** *Differentiable Unsupervised Feature Selection* (Lindenbaum et al.,
  NeurIPS 2021).  Here its continuous gates define a sample-space metric; they
  do not hard-delete features.
- **L:** Laplacian regularization of the fused sample scores.
- **IU-PCR:** the uncorrelated-error variant of **U-PCR**, *Unsupervised
  Principal Component Regression* (Tenzer et al., AISTATS 2022).  The paper
  denotes this variant IU-PCR and defines it through pairwise-uncorrelated
  expert deviations.  Repository prose may call it the independent-error
  variant, but the mathematical assumption is pairwise uncorrelated errors,
  not full statistical independence.

Do not use **TA-LIU** as a synonym.  TA-LIU is the later target-anchored,
label-using diagnostic and is outside the strict label-free research direction.

## Mathematical definition

Let $F\in\mathbb{R}^{m\times n}$ be the centered feature matrix and
$C=FF^\top/n$.  Ordinary IU-PCR estimates the unavailable vector
$\rho_i=\operatorname{Cov}(f_i,Y)$ from the unlabeled off-diagonal covariance
structure.  If $U$ contains the leading two eigenvectors of $C$, its weights
are

\[
w_0=U(U^\top C U)^{-1}U^\top\hat\rho.
\]

DUFS learns continuous feature gates $g$.  A self-tuning nearest-neighbour
graph is constructed from the gated sample coordinates
$z_j=g\odot F_{:j}$, and $L$ is its symmetric normalized graph Laplacian.
For fused scores $s=F^\top w$, graph roughness is

\[
\frac{1}{n}s^\top Ls=w^\top Rw,
\qquad R=\frac{1}{n}FLF^\top.
\]

After trace-matching $R$ to $C$ inside the same two-dimensional PCR
subspace, DUFS-LIU changes only the final IU-PCR weight solve:

\[
w_\lambda=
U\left[U^\top(C+\lambda\bar R)U\right]^{-1}U^\top\hat\rho.
\]

At $\lambda=0$, it reproduces ordinary IU-PCR exactly.  No labels enter the
gate learner, graph, covariance estimate, $\hat\rho$, or weight solve.

## What remains useful

The frozen synthetic study demonstrated a DUFS-specific positive mechanism:
on the smooth-signal world DUFS-LIU improved IU-PCR by **+0.382 +/- 0.149 AUROC
points** and had positive paired lower bounds versus ungated, shuffled-gate,
permuted-graph, and projected-ridge controls.  This is the closest implemented
realization of the advisor proposal to use DUFS's differentiable spectral
mechanism to improve U-PCR parameter estimation rather than merely perform hard
feature selection.

It is not a successful detector extension. It missed the registered
+0.5-point mechanism threshold and lost **-0.568 +/- 0.049 points** when DUFS
learned a nuisance manifold.  The open research problem is therefore:

> Preserve the label-free DUFS-to-U-PCR parameter-estimation mechanism while
> preventing an unlabeled but smooth nuisance geometry from controlling the
> Laplacian penalty.

The full 24-cell view-fusion benchmark later showed the same broader problem:
even stable micro-view geometry was harmful, and sample-specific reliability
did not beat global or permuted controls. The next design therefore removes
local alpha and learned clustering. Before any learner is built, the Phase-0
atomic-operator audit must test whether a frozen label-free diagnostic predicts
which atomic Laplacians help IU-PCR across held-out families. If it does not,
the graph-regularization line closes. See
`docs/research_notes/atomic_operator_gating_plan.md`.

## Source of record

- Implementation: `spectral_utils/laplacian_upcr.py`
- Synthetic experiment: `scripts/laplacian_upcr_synthetic.py`
- Frozen result: `results/laplacian_upcr_synthetic/REPORT.md`
- Full real-data conclusion:
  `docs/research_notes/frozen_24cell_view_fusion_conclusion.md`
- Stopped safety detour: `results/graybox_cross_view_phase1/CONCLUSION.md`
