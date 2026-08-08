# DUFS-LIU: DUFS-gated Laplacian-regularized IU-PCR

## Terms used below

A **gate** is a number attached to a feature. A large gate lets that feature
strongly affect distance between samples; a small gate suppresses it. A
**nearest-neighbour graph** connects samples with small gated distance. Its
**Laplacian** (L) measures score roughness on those connections. (lambda)
controls how strongly roughness changes the IU-PCR weights.

## Paper basis and claim boundary

The gate learner is based on Lindenbaum et al.,
[Differentiable Unsupervised Feature Selection based on a Gated Laplacian](https://proceedings.neurips.cc/paper/2021/hash/0bc10d8a74dbafbf242e30433e83aa56-Abstract.html),
NeurIPS 2021. The IU-PCR reliability estimate is based on Tenzer et al.,
[Crowdsourcing Regression: A Spectral Approach](https://proceedings.mlr.press/v151/tenzer22a.html),
AISTATS 2022.

The source papers do not contain DUFS-LIU. DUFS originally selects features for
unsupervised structure such as clustering. Our method uses its continuous gates
to define a metric and then changes the final IU-PCR weight equation.

## Current feature contract

The original frozen 24-cell benchmark used `fixed_stable_v1`, which removed
`pe_mean`, `stft_spectral_entropy`, `cusum_shift_idx`, and `rpdi`. A later
development search tested all 256 combinations of four operations for those
features: remove, raw, squared, and mode-centred.

The next-run candidate is
`dufs-liu-mixed-v2-development-2026-08-07`:

| feature | frozen operation |
|---|---|
| `pe_mean` | `-z^2` |
| `stft_spectral_entropy` | `-|rank(x)-mode_rank|` |
| `cusum_shift_idx` | confidence-oriented raw value |
| `rpdi` | confidence-oriented raw value |

The transformed column replaces its raw parent. The two copies are never used
together. The mode is estimated from the unlabeled feature distribution inside
the cell by KDE. Missing features remain missing.

This mapping was selected on the existing 24 development cells. It is frozen
for the next external run, but it is not confirmed. Its retrospective DUFS-LIU
AUROC is 0.776562, compared with 0.774139 for stable-only. LOFO selection gave
+0.123pp, but most of that estimate came from one MATH-500/Qwen cell. See
`docs/research_notes/dufs_liu_mixed_feature_contract_conclusion.md` for the
selection boundary.

## DUFS gate learning

For feature (r), DUFS uses a stochastic hard-sigmoid gate

\[
z_r=\min(1,\max(0,\mu_r+\epsilon_r)),
\qquad \epsilon_r\sim\mathcal N(0,\sigma^2). \tag{1}
\]

The probability that the gate is active is

\[
p_r=\Pr(z_r>0)=\Phi(\mu_r/\sigma). \tag{2}
\]

At each training step, the features are multiplied by sampled gates, a
self-tuning sample graph is recomputed in that gated space, and a diffusion
smoothness score is optimized. We use the paper's parameter-free form (its
Equation 7), implemented as

\[
\mathcal L_{\mathrm{DUFS}}
=-
\frac{\operatorname{tr}(\widetilde X^\top P^t\widetilde X)/B}
{\sum_r p_r+\delta}, \tag{3}
\]

where (B) is batch size, (P) is the random-walk matrix of the current gated
graph, and (t) is its diffusion power. The sign in Equation (3) means that
minimization rewards gated features that are predictable from their graph
neighbours while the denominator discourages keeping every feature.

Three random seeds produce (p_r). Their mean is RMS-normalized to a distance
gate (g_r). No threshold is applied and no feature is deleted.

## Graph and LIU fusion

For sample (j), define gated coordinates

\[
z_j=g\odot F_{:j}. \tag{4}
\]

A symmetric self-tuning (k)-nearest-neighbour graph (W) is built from these
coordinates. Its normalized Laplacian is

\[
L=I-D^{-1/2}WD^{-1/2}. \tag{5}
\]

For a fused score (s=F^\top w),

\[
\frac1n s^\top Ls=w^\top Rw,
\qquad R=\frac1n FLF^\top. \tag{6}
\]

Let (U) be the same two-dimensional IU-PCR subspace. We trace-match the
projected roughness to the projected covariance, producing (ar R), and solve

\[
w_\lambda=
U\left[U^\top(C+\lambda\bar R)U\right]^{-1}
U^\top\hat\rho. \tag{7}
\]

At (lambda=0), Equation (7) must equal ordinary IU-PCR exactly.

## Assumptions

1. Features that preserve a useful low-frequency sample structure should receive
   larger DUFS gates.
2. Local neighbours in the gated feature space should have similar correctness
   scores, even though correctness is not used to build the graph.
3. Useful graph roughness is visible inside IU-PCR's two-dimensional subspace.
4. Trace matching makes one (lambda) reasonably comparable across cells.

The important failure is a smooth nuisance manifold: DUFS can correctly learn
an unlabeled structure that is unrelated or opposed to correctness. A stable
DUFS loss is therefore not proof that the graph is useful for detection.

## Frozen hyperparameters

| parameter | value | role |
|---|---:|---|
| DUFS seeds | 11, 23, 37 | estimate gate stability |
| DUFS epochs | 80 | optimization length |
| DUFS loss | parameter-free Eq. 7 | avoids a sparsity coefficient selected with labels |
| final graph neighbours | 7 | sample locality |
| LIU lambda | 0.1 | registered DUFS-LIU headline strength |
| LIU subspace | 2 PCs | same as ordinary IU-PCR |
| feature contract | mixed-v2 development candidate | frozen after the 256-contract search |

The report also shows a fixed lambda sensitivity path, but does not replace
the headline setting with the best observed value.

## Checks required

- DUFS effective feature count and per-feature gate probabilities;
- gate variation across seeds and near-zero/near-one fractions;
- graph components, degree quantiles, edge mass, and spectral gap;
- exact (lambda=0) identity;
- projected roughness eigenvalues and system condition number;
- weight cosine, score rank displacement, and Laplacian energy versus IU-PCR;
- comparison with an ungated raw-feature graph;
- AUROC, AUPRC, paired uncertainty, wins/losses, and worst-cell loss.

## Computational cost

With (T) epochs, batch size (B), and (m) features, the current DUFS
training uses pairwise batch affinities and costs roughly
(O(TB^2m)) per seed. The final neighbour search is implementation-dependent;
its worst case is (O(n^2m)), while the stored sparse graph has (O(nk))
edges. Computing (R=FLF^\top/n) costs about (O(mnk+m^2n)) with a sparse
graph. The final two-dimensional solve is negligible.
