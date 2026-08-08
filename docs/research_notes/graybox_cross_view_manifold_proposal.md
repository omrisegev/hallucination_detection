# Gray-Box Cross-View Manifold Identification for Laplacian IU-PCR

## Status and purpose

Status: **revised after the U2-prior reconciliation; executable Phase-1 plan and
contingent Phase-2 design, not yet implemented**

This document is a research and experiment handoff. It defines the proposed
algorithm, the motivation from prior repository results, the allowed information
boundary, the checks that must precede implementation, the synthetic evaluation,
the failure criteria, and the conditions for later evaluation on real
hallucination data.

This revision incorporates the latest repository result: a current IU-prior
two-direction head is a reparameterized prior-bearing head in the ordinary U2
score space on all eight paired synthetic matrices, while the historical
`anchored_pcr2` basis is different under its historical configuration. The
three saved real U2/anchored heads are negative on average, but a literal
cell-level stop rule remains open because `anchored_pcr6` gains 1.491 points on
one cell. That is a localized heterogeneity question, not evidence for another
global head correction.

Accordingly, this proposal freezes the fusion head and tests a different lever:
whether a graph is safe enough to regularize IU-PCR. It does not claim that the
U2 family is mathematically closed, and it does not compete with the still-open
few-label subset-adaptation direction.

Implementation must proceed in two user-visible checkpoints. Phase 1 tests the
cross-view audit premise on low-dimensional synthetic views without a trajectory
bank or a new DUFS optimizer. Phase 2 builds the minimal primitive-disjoint
trajectory pipeline only if Phase 1 passes. Stop for discussion after each
phase. Do not open confirmation seeds or real hallucination data automatically.

This is a proposal, not an established result. The central hypothesis must be
falsified synthetically before any claim is made about hallucination detection.

## Executive summary

The current Laplacian IU-PCR candidate uses one feature matrix for three roles:

1. DUFS learns continuous feature gates from it.
2. The gated coordinates construct a sample-neighborhood graph.
3. The same feature matrix is regularized and fused by IU-PCR.

This creates a same-view loop. A graph optimized to make the input coordinates
locally smooth is then evaluated through the smoothness of those same
coordinates. The Phase-1 synthetic study proved that this mechanism can find a
real geometry, but it cannot determine whether that geometry represents
correctness or a cleaner nuisance variable.

The proposed method separates three roles:

- **Discovery features** build a candidate graph directly in Phase 1 and through
  group-gated adapted DUFS in Phase 2.
- **Held-out audit features** test whether that geometry transfers to a distinct
  probability-derived view and is not merely a length/truncation manifold.
- **Fusion features** remain a frozen stable set used by IU-PCR.

A transformation bank is not part of the Phase-1 premise test. In Phase 2, a
small frozen bank is permitted only in the graph branch. It must be normalized
and gated at the primitive-group level so that one primitive is not counted many
times because more transformations were generated for it. The fusion branch is
frozen and identical between the baseline and candidate.

The intended contribution is:

> Label-free cross-view identification and safety testing of a candidate
> probability-trajectory manifold before it is allowed to regularize IU-PCR.

The proposed method does **not** claim that several transformations of the same
token probabilities become independent correctness estimators. All views still
come from the same gray-box observation and may share a nuisance.

### Review corrections incorporated here

The original draft had eight blocking ambiguities:

1. “Temporal” and “distribution” families reused entropy, margin, and tail
   primitives, making held-out transfer partly deterministic. The primary split
   is now primitive-disjoint.
2. Raw normalized-Laplacian energy is not comparable across graphs or `k`.
   Transfer is now calibrated against degree/spectrum-preserving node
   permutations and reported as an effect size and p-value.
3. The current fixed-stable pool contains raw-logit energy views that violate
   the strict normalized-probability contract. The strict-graybox fusion pool
   explicitly removes them.
4. One shared gate per primitive was requested without defining its objective.
   Phase 1 therefore uses no learned gates; Phase 2 defines the group-gated
   adaptation explicitly and tests exact-duplicate invariance before use.
5. The earlier 10-world/20-arm design mixed premise testing, representation
   engineering, and estimator selection. The experiment is now staged.
6. Cross-family holding out is not statistical cross-fitting. This document now
   calls it **out-of-family auditing**; sample cross-fitting is reserved for
   nuisance residualization.
7. “Zero-label” could be read as an end-to-end label-free history, although the
   fixed orientation and stable-pool contracts were informed by prior labeled
   development audits. The claim is now explicitly target-cell-zero-label and
   conditional on a frozen fusion contract.
8. Primitive-disjoint does not mean statistically independent: when
   \(y_t\sim p_t\), expected selected-token surprisal equals predictive entropy.
   A new unmeasured-shared-nuisance world tests this sampling-identity blind
   spot explicitly.

## 1. Hard information-access contract

### 1.1 Allowed inputs

The core method may use only gray-box generation outputs derived from normalized
token distributions:

- the normalized token distribution \(p_t(v)\), when available;
- top-\(K\) token log-probabilities at each generation step;
- the generated token's probability or log-probability;
- generated token IDs, solely to align the selected probability with the
  generated token;
- sequence length, EOS position, truncation status, and validity masks;
- deterministic transformations and temporal summaries of the quantities above.

Labels may be used only by the experiment evaluator after every method output is
frozen. Synthetic latent variables may be used only for generating a world and
constructing a clearly marked oracle ceiling.

### 1.2 Disallowed inputs

The core method must not use:

- hidden states or embeddings;
- attention maps;
- gradients, Jacobians, NTKs, or layer-wise activations;
- model-weight or activation noise injection;
- latent interventions;
- raw question/answer text, semantic embeddings, retrieval, or external evidence;
- LLM-as-judge scores;
- human labels, calibration labels, pseudo-labels, verbalized correctness labels,
  or answer-agreement labels;
- raw-logit magnitude or log-partition information that is not recoverable from
  the normalized token distribution.

If a later extension wants to admit raw unnormalized logits, it must be a
separate access tier and a separate experimental arm. It must not silently enter
the strict gray-box result through `token_logsumexp`, Semantic Energy, or an
absolute-energy feature.

### 1.3 Top-K distributions

When only top-\(K\) probabilities are saved, define the residual tail mass as

\[
r_t = 1-\sum_{k=1}^{K}p_{t,(k)}.
\]

The residual may be treated as one aggregate tail bin. This gives a coarsened
distribution, not the missing tail shape. Entropy, Renyi-2, effective support,
and varentropy computed this way must be named lower-bound/coarsened
approximations as appropriate; they are not exact full-vocabulary quantities.
Existing exact `token_entropies` may be used when their provenance is verified.

The repository's `top_k_logprobs` stores full-distribution log-probabilities for
the retained top-K tokens, so residual mass is recoverable. Exact
selected-token surprisal is available only when provenance verifies either an
aligned normalized-probability trace (currently `token_spilled_energies`) or the
generated token is present with its probability. `gen_token_ids` alone does not
recover a probability when the generated token falls outside the saved top-K.
Such rows must be marked missing or bounded; they must not be silently assigned
the K-th probability.

### 1.4 Epistemic status of the frozen fusion contract

The candidate uses no labels from the target cell and no labels inside graph
construction, auditing, gating, or scoring. However,
`CONFIDENCE_FEATURE_SIGNS_V1`, `FIXED_STABLE_EXCLUDED_V1`, and GOOD_6 were
informed by earlier labeled development analyses. They are frozen historical
artifacts shared by the candidate and its IU-PCR baseline, not newly learned in
this experiment.

Therefore the defensible contribution is **zero-label graph adaptation
conditional on a development-fixed fusion contract**, not an end-to-end method
whose complete research history never used labels. Report that distinction.
The paired candidate-minus-IU comparison still isolates the graph because both
arms receive the identical oriented fusion matrix. A fully prior-free
orientation experiment is a separate question and may not be mixed into this
one.

## 2. Prior repository evidence that constrains the proposal

The next implementation must begin by reproducing or reading the following
results. They prevent this project from repeating closed directions.

### 2.1 HMM, AR, Kalman, BOCPD, and anomaly scoring

`HISTORY.md`, Step 151, tested:

- a two-state Gaussian HMM;
- BOCPD;
- AR(2) residual error;
- Kalman innovations/NIS;
- Mahalanobis, GMM, KDE, Isolation Forest, autoencoder, and PRAE-style scores.

On the clean GSM8K/Llama-8B pilot:

- `hmm_occ`: 0.719 AUROC;
- `ar2_mse`: 0.717;
- `kalman_nis`: 0.703;
- DeepConf: 0.735;
- L-SML GOOD_5: 0.754.

AR/Kalman innovations correlated approximately 0.93--0.97 with oriented EPR,
showing that they mostly repackaged entropy level. The fitted high-entropy HMM
state was not sticky enough to support the proposed hallucination-momentum
mechanism. BOCPD was the useful exception: its change-point score was nearly
orthogonal to entropy level, but did not improve the exploratory six-view
fusion.

Step 153 then evaluated these derived views more broadly. Adding them as fusion
features was harmful:

- anomaly views: roughly -4.9 to -7.9 AUROC points;
- HMM/AR/Kalman: roughly -3.1 to -7.4 points;
- BOCPD: roughly -4.8 points;
- BOCPD on the selected-token log-probability trace was useful alone in one
  cell, but harmful when fused.

Therefore:

> HMM, AR, Kalman, and BOCPD are closed as default additional fusion features.
> They may be revisited only as graph-only coordinates or explicit ablations in
> the new cross-view design.

### 2.2 More individually informative features did not imply additive value

Step 181 found cell-specific gains from `cusum_max_spilled` and
`topk_tail_mass`. The later replication-grid experiment showed that the former
did not replicate across cells, while `varentropy` did generalize and became the
sixth member of GOOD_6.

Step 206 then tested the add/remove lever directly. Adding strong
`topk_tail_mass` and `renyi_entropy_2` variants to GOOD_6 did not improve it.
The shared finding was:

> High individual informativeness is not the same as additive information.
> Several readings of one token distribution can occupy the same signal
> direction and harm a dependency-sensitive fusion estimator.

This is why the proposed large transformation bank belongs in the geometry
branch, not automatically in the IU-PCR fusion matrix.

### 2.3 DUFS currently optimizes a criterion weakly related to separability

The selector analysis found that the correlation between DUFS gate value and a
feature's oriented AUROC was only about 0.149 on average. Selected features were
better on average than unselected ones, but the relationship was too weak to
recover the compact supervised-by-development GOOD_6 subset.

This is not an optimizer defect. DUFS optimizes local smoothness/reconstruction,
not class separability. The new experiment must not use a same-view smoothness
value as evidence that the discovered geometry represents correctness.

### 2.4 The existing Laplacian IU-PCR result

`results/laplacian_upcr_synthetic/REPORT.md` established:

- smooth signal world: +0.382 +/- 0.149 AUROC points over IU-PCR;
- correlated-error world: -0.009 +/- 0.008;
- nuisance manifold: -0.568 +/- 0.049;
- the positive gain was below the frozen +0.5-point meaningful-effect gate;
- the nuisance failure violated the safety gate.

The learned DUFS gates correctly found a clean geometry. They could not identify
whether it was the target geometry. This is the specific failure the new method
must address.

### 2.5 The latest U2 and target-anchored results

`results/target_anchored_laplacian_synthetic/DEVELOPMENT_REPORT.md` showed that
16 labels could switch graph gates to the planted target block and repair the
nuisance failure. However, TA-LIU used those labels much less effectively than
ordinary U2 logistic on the target-g world: `+1.267` versus `+19.523` AUROC
points over IU-PCR. Label injection proved target identifiability, not an
advantage for Laplacian fusion.

`results/u2_prior_reconciliation/REPORT.md` then established:

- current IU-prior logistic spans ordinary U2 exactly on 8/8 paired matrices;
- 1,280/1,280 fitted reparameterizations agree within frozen tolerances;
- historical `anchored_pcr2` does not span full-matrix U2 under its historical
  exclusion/recompute/fallback/component configuration;
- at 20 labels, saved real `gold_pcr2`, `anchored_pcr2`, and `anchored_pcr6`
  average `-4.279`, `-0.149`, and `-0.355` points versus U-PCR;
- one `anchored_pcr6` cell (`math500_qwenmath7b`) has `+1.491` points of
  cell-specific headroom, so the literal per-cell stop condition is false.

These results constrain the contribution. The proposed method must keep the
ordinary IU-PCR covariance, rho estimate, and U2 head fixed. Any gain must be
attributed to graph identification and safety, not to another low-dimensional
logistic or prior variant. The isolated positive cell remains a separate
heterogeneity diagnostic and is not a reason to tune this zero-label method.

### 2.6 Closest spectral multi-view precedent

The closest local paper is Yang, Lindenbaum, Kluger, and Jaffe,
*Multi-modal Differentiable Unsupervised Feature Selection* (arXiv:2303.09381).
It constructs a symmetric shared operator

\[
P_{\mathrm{shared}}=L_XL_Y+L_YL_X
\]

and learns stochastic gates in both registered modalities. This is more direct
precedent than ordinary DUFS for the claim that structure shared across views can
be separated from view-specific structure.

It does not solve our target-identification problem: a nuisance shared by every
probability view also appears in its shared operator. Its normalized-affinity
operator also differs from this repository's `I-D^{-1/2}WD^{-1/2}` penalty, and
the product is not inserted directly into the IU-PCR roughness equation. Phase 1
therefore includes a clearly named **mmDUFS-inspired shared-affinity control**:
form the paper's symmetric product from the two normalized affinities, treat its
nonnegative symmetric entries as a new affinity, and construct a fresh normalized
Laplacian before IU-PCR. This is an adaptation/control, not a paper-faithful
mmDUFS implementation.

## 3. Scientific hypothesis

### 3.1 Primary hypothesis

A candidate graph learned from one primitive family is more likely to encode a
reusable underlying process when its neighborhood structure transfers to a
**primitive-disjoint** audit family that was not used to build the graph.

The falsifiable claim is not merely that the transfer statistic is lower. A
permutation-calibrated out-of-family audit must accept most shared-signal graphs,
reject family-specific nuisance graphs and pure-noise graphs, and make the final
Laplacian method safer than same-view DUFS-LIU without losing its positive
mechanism.

### 3.2 Safety hypothesis

A graph dominated by a known gray-box nuisance such as length, truncation, or
nearly deterministic-token rate can be detected through alignment with a
nuisance-only graph. A graph that is not stable under resampling or small changes
in neighborhood size should also be rejected.

### 3.3 Explicit limitation

Cross-view agreement does not prove correctness. All feature families are
functions of the same token distributions. A strong latent nuisance may affect
all of them.

There is also a direct probabilistic link between the two registered views. If
the generated token \(Y_t\) is sampled from \(p_t\), then

\[
\mathbb E[-\log p_t(Y_t)\mid p_t]=H(p_t).
\]

Thus selected-token surprisal can validate an entropy geometry even when the
shared latent variable is only generic uncertainty, temperature, or difficulty.
Primitive-disjoint auditing removes deterministic transformation leakage; it
does not create independent evidence. P1-F and P2-F below are mandatory tests of
this limitation.

Nor does failure imply that richer trajectories are useless. It rejects the
specific claim that out-of-family graph transfer is a sufficient zero-label
safety signal for this fusion mechanism.

The valid claim, if the method succeeds, is narrower:

> Cross-view auditing reduces harmful use of self-confirming or known-nuisance
> probability manifolds while preserving useful Laplacian regularization.

Do not claim universal unsupervised identification of correctness.

## 4. Mathematical definition

Assume \(n\) generated answers. Use samples as rows for the definitions below.

- \(G\in\mathbb{R}^{n\times d_G}\): discovery coordinates.
- \(A\in\mathbb{R}^{n\times d_A}\): held-out audit coordinates.
- \(N\in\mathbb{R}^{n\times d_N}\): known nuisance coordinates.
- \(F\in\mathbb{R}^{n\times m}\): frozen fusion features.

All preprocessing must be label-free and fitted within the current dataset/cell.

### 4.1 Discovery graph

Phase 1 standardizes the columns of the low-dimensional synthetic discovery
view and constructs a self-tuning symmetric k-NN graph directly. It deliberately
uses no DUFS and no transformation bank; otherwise failure of the audit would be
confounded with gate optimization.

Phase 2, if reached, uses the frozen group-gated adapted-DUFS construction in
Section 6.3 to obtain nonnegative primitive-group gates \(a\). Construct the
graph from the gated coordinates:

\[
W_G(i,j)
=
\exp\left(
-\frac{\lVert a\odot G_i-a\odot G_j\rVert^2}
{\sigma_i\sigma_j}
\right),
\]

followed by the symmetric normalized Laplacian

\[
L_G=I-D^{-1/2}W_GD^{-1/2}.
\]

Reuse \`self_tuning_knn_graph\` and \`symmetric_normalized_laplacian\` from
\`spectral_utils/laplacian_upcr.py\`. Do not substitute the nonsymmetric DUFS
random-walk matrix. The repository implementation is an adapted DUFS optimizer,
not the paper's exact optimizer, and must be named accordingly.

### 4.2 Cross-view transfer diagnostic

For each standardized audit coordinate \(a_j\), define its graph energy

\[
E_j(L_G)=\frac{a_j^\top L_Ga_j}{a_j^\top a_j+\epsilon}.
\]

Raw energy is not comparable across graphs with different degree distributions,
connectivity, or \(k\). Calibrate each coordinate against \(B=199\)
deterministic node permutations of the same graph. \`permute_graph\` preserves
the graph's spectrum, degrees, and edge-weight multiset while breaking its
alignment to the audit rows. Define

\[
Z_j=
\frac{\operatorname{median}_b E_j(P_bL_GP_b^\top)-E_j(L_G)}
{1.4826\,\operatorname{MAD}_b E_j(P_bL_GP_b^\top)+\epsilon}.
\]

Positive \(Z_j\) means the observed graph makes the audit coordinate smoother
than its permutation null. Report the exact empirical one-sided p-value
\((1+\#\{E_{\mathrm{perm}}\leq E_{\mathrm{obs}}\})/(B+1)\). Aggregate
coordinates within an audit primitive by their median \(Z_j\), then aggregate
primitives by the median again; never pool every transformation as an
independent vote. Let the resulting value be \(T_{\mathrm{obs}}\). Apply the
same two-level aggregation to every synchronized permuted pseudo-observation to
obtain \(T_b\), and define the graph-level audit p-value as

\[
p_{\mathrm{audit}}=
\frac{1+\#\{b:T_b\geq T_{\mathrm{obs}}\}}{B+1}.
\]

This single graph-level p-value, rather than the coordinate p-values, drives the
accept/reject rule in Section 11.3.

Compare this calibrated statistic with:

- the ungated/direct discovery graph;
- a length/truncation-only graph;
- the mmDUFS-inspired shared-affinity control;
- graphs rebuilt under gate seeds and nearby \(k\) values;
- audit-row permutation, which must destroy the transfer signal.

The **primary candidate is a hard veto**. Weighted consensus is deferred until a
hard-veto rule passes the Phase-1 premise test; it may not be searched in
parallel and substituted after seeing performance.

### 4.3 Nuisance alignment

Build a nuisance graph \(W_N\) from standardized allowed metadata: normalized
length, EOS position, truncation, incomplete-top-K fraction, and transform-valid
counts. Measure graph similarity with centered kernel alignment (CKA) between
dense centered affinity matrices. At the planned synthetic size (\(n=360\))
this is tractable and avoids an edge-support correlation dominated by whichever
k-NN edges happen to overlap. Calibrate nuisance CKA against the same
node-permutation scheme.

Two nuisance diagnostics are required:

1. raw length/truncation coordinates;
2. transfer to audit residuals after five-fold, row-cross-fitted ridge regression
   of each audit coordinate on nuisance coordinates. Folds and ridge strength are
   fixed without labels.

Residualization is a diagnostic, not the primary score path. It is not
automatically safe because correctness may genuinely correlate with length. A
candidate graph is vetoed when its raw nuisance alignment is significant and its
out-of-family transfer disappears after nuisance residualization under the
frozen Phase-1 rule.

### 4.4 Laplacian IU-PCR

Let the fusion matrix in repository orientation be
\(F_c\in\mathbb{R}^{m\times n}\). For an accepted graph, compute

\[
R_{F\mid G}=\frac{1}{n}F_cL_GF_c^\top.
\]

Keep the ordinary IU-PCR covariance \(C_F\), target-covariance estimate
\(\hat\rho_{\mathrm{IU}}\), and two-dimensional spectral subspace \(U\). Only
the final projected solve changes:

\[
w_\lambda
=
U\left[U^\top(C_F+\lambda\bar R_{F\mid G})U\right]^{-1}
U^\top\hat\rho_{\mathrm{IU}}.
\]

Use the existing trace-matching rule for \(\bar R\). At \(\lambda=0\), the
score and weights must reproduce ordinary IU-PCR exactly.

### 4.5 Cross-view rotations

Do not make one direction of the primitive split the complete method. Evaluate
both registered directions:

- distribution-state primitives discover, realized-token primitives audit;
- realized-token primitives discover, distribution-state primitives audit.

All temporal transformations of one primitive stay on the same side. A
consensus graph combines only directional graphs that pass out-of-family
transfer and nuisance checks. Other rotations are exploratory and require a new
registered primitive-source split; they may not be created after seeing which
coordinates transfer.

This is out-of-family auditing, not sample cross-fitting. If no graph passes,
the method returns the ordinary IU-PCR score exactly.

### 4.6 Frozen primary candidate shape

The Phase-2 primary candidate has two registered modalities:

- **distribution-state view**: predictive entropy, top-1 probability, top-two
  margin, tail mass, Renyi-2 concentration, and effective support;
- **realized-token view**: exact selected-token surprisal and
  selected-versus-top-1 gap, only where provenance makes them exact.

Each view learns one graph. Each graph is audited only on the other view and on
the nuisance diagnostics. Accepted graphs are combined by an unweighted mean of
their affinity matrices, followed by reconstruction of a symmetric normalized
Laplacian. If neither passes, return IU-PCR. If only one passes, use only that
graph. Persist the asymmetry and fallback reason.

The primary uses the previously frozen \(k=7\) and \(\lambda=0.1\) so graph
quality is isolated from retuning regularization. \(k\in\{5,7,11\}\) and the
existing lambda path are diagnostics only in the first development run.

## 5. Primitive gray-box trajectories

Construct primitive time series before constructing sample-level feature banks.
At each generation step \(t\), candidates include:

1. Predictive entropy

   \[
   H_t=-\sum_v p_t(v)\log p_t(v).
   \]

2. Selected-token surprisal

   \[
   S_t=-\log p_t(y_t).
   \]

3. Maximum probability \(p_t^{(1)}\).

4. Top-two margin

   \[
   M_t=p_t^{(1)}-p_t^{(2)}.
   \]

5. Top-K tail mass or residual mass.

6. Collision/Renyi-2 concentration

   \[
   C_t=\sum_v p_t(v)^2,
   \qquad H_{2,t}=-\log C_t.
   \]

7. Effective support size, such as \(\exp(H_t)\) or \(1/C_t\).

8. Token-level varentropy when it can be computed faithfully. Top-K
   renormalized varentropy is a different, coarsened feature and must carry a
   different manifest name.

9. Optional cross-channel quantities such as entropy-margin disagreement.

Every primitive must record whether it is exact, top-K approximated, or derived
from a residual-tail bin. The primary out-of-family split is by primitive
source, not by temporal transformation:

- distribution-state primitives: items 1 and 3--8;
- realized-token primitives: item 2 and its exact gap from top-1;
- nuisance primitives: length, truncation, incomplete capture, and validity.

Selected-token rank is not primary because it is right-censored when the sampled
token lies outside saved top-K. A capped-rank/missingness representation may be
reported as an ablation only. Temporal mean, slope, DCT, CUSUM, and segment
summaries of the same primitive must remain in the same view; moving a mean to
the audit family while its DCT coefficients remain in discovery would create a
deterministic audit leak.

The distribution-state/realized-token split is therefore **primitive-disjoint,
not independent**. The entropy--surprisal conditional-expectation identity in
Section 3.3 must be carried into the null worlds and the final claim.

## 6. Transformation bank

The purpose of the transformation bank is to provide richer coordinates for
manifold discovery. It is not evidence that independent verifiers were created.

### 6.1 Preferred transformations

Normalize token position to \(u=t/T\in[0,1]\). The Phase-2 primary dictionary is
deliberately small:

- whole-trace mean and standard deviation;
- four equal-position-bin means, with a validity mask;
- linear slope;
- maximum positive and negative first difference;
- total variation;
- the first three non-DC DCT coefficients after deterministic resampling to 32
  positions.

This is 13 coordinates per primitive before constant-column removal. It tests
level, coarse shape, local change, and low-frequency structure without searching
dozens of transforms. Robust quantiles, 8/16-bin profiles, extrema locations,
drawdown/rebound, autocorrelation, cross-channel correlations, CUSUM, and
change-point models are deferred ablations. Cross-channel transforms form mixed
primitive groups and must never be assigned to only one side of the primary
out-of-family audit.

Prefer normalized segments or low-order DCT to STFT for short answers. Earlier
STFT features were weak, resolution-limited on short traces, and in one case
highly temperature-sensitive.

### 6.2 Secondary ablations only

The following may be included only as named ablations, not as required parts of
the primary candidate:

- two-state HMM occupancy and transition summaries;
- BOCPD event probability and change location;
- AR residual error;
- Kalman innovations;
- high-order FFT/STFT features;
- generic polynomial expansion of final scalar summaries.

Their prior fusion failures must be reported next to any new graph-only result.

### 6.3 Group normalization and shared gates

Suppose primitive signal \(j\) produces a transformation group

\[
\Phi_j=[\phi_{j1},\ldots,\phi_{jr_j}].
\]

Normalize the group so that the total scale assigned to a primitive does not
grow with the number of transformations:

\[
\widetilde\Phi_j
=
\frac{\Phi_j}
{\sqrt{\sum_k\operatorname{Var}(\phi_{jk})}+\epsilon}.
\]

The Phase-2 preferred candidate uses one stochastic gate \(z_j\) per primitive
group and applies it to every column of \(\widetilde\Phi_j\). Let
\(X_z=[z_1\widetilde\Phi_1,\ldots,z_q\widetilde\Phi_q]\), and let
\(P_z^2\) be the two-step random walk constructed from \(X_z\), matching the
repository's adapted-DUFS implementation. The frozen group adaptation minimizes

\[
\mathcal L_{\mathrm{group}}
=
-\frac{\operatorname{tr}(X_z^\top P_z^2X_z)}
{n\left(\sum_{j=1}^q\Pr[z_j>0]+\epsilon\right)}.
\]

The denominator counts groups, not coordinates. Use the existing stochastic-gate
parameterization and frozen optimizer constants: \(\sigma=0.5\),
\(\mu_0=0.5\), learning rate \(0.02\), batch cap 256, 120 epochs, and seeds
\`(11,23,37)\`. This is a new group adaptation of the repository optimizer, not
a claim about the original DUFS paper. Per-coordinate adapted DUFS is a control.
Phase 1 does not use either optimizer.

### 6.4 Duplication invariance requirement

After group normalization, duplicating one primitive's transformation columns
must not materially change:

- its total group norm;
- graph edge weights;
- accepted/rejected graph status;
- the final score.

Create a unit test that duplicates one group 2x, 5x, and 10x. This test must pass
before any performance run. Freeze tolerances before execution: total group
squared-distance relative error at most \(10^{-10}\), graph maximum edge-weight
error at most \(10^{-6}\), identical accept/reject decisions, and final
standardized-score maximum error at most \(10^{-6}\). Constant groups are removed
with a recorded reason before gate learning.

## 7. Initial feature-family proposal

The split is determined by what random quantity is observed at a token, not by
real AUROC and not by the later transformation applied to it.

### 7.1 Distribution-state view

Candidates:

- predictive entropy;
- top-1 probability;
- top-two margin;
- residual/top-K tail mass;
- Renyi-2 concentration and effective support;
- varentropy, only under the provenance-specific exact or coarsened name.

Every allowed temporal transform of these primitives belongs to this view.
For example, entropy mean, entropy bins, and entropy DCT coefficients may not be
split across discovery and audit.

### 7.2 Realized-token view

Candidates:

- exact selected-token surprisal;
- selected-token versus top-1 surprisal gap.

Both require a verified normalized probability for the generated token.
Selected rank is an ablation because it is censored outside the saved top-K.
This view is smaller by design; adding entropy or margin transforms to make it
larger would invalidate the audit.

### 7.3 Nuisance family

Candidates:

- trace length;
- EOS-relative position;
- truncation indicator and distance to generation cap;
- fraction of near-deterministic tokens;
- fraction of steps with incomplete top-K mass;
- number of valid windows/segments for temporal transforms.

### 7.4 Frozen fusion family

Define the primary fusion pool as

```text
STRICT_GRAYBOX_FIXED_STABLE_V1 =
    FIXED_STABLE_POOL
    - {epr_energy, min_energy, sw_var_peak_energy, cusum_max_energy}
```

and retain only coordinates whose manifest proves derivation from normalized
probabilities or allowed metadata. The four exclusions above use the raw
log-partition series and therefore violate Section 1 even though they are part
of the repository's current `FIXED_STABLE_POOL`. Availability filtering,
constant-column removal, orientation, and standardization must otherwise match
the current feature contract exactly.

The primary baseline is ordinary IU-PCR on this strict pool. Report two
secondary references without allowing either to select the new method:

- current deployed IU-PCR on the full fixed-stable pool, clearly marked as a
  broader raw-logit access tier;
- GOOD_6 (`epr`, `low_band_power`, `sw_var_peak`, `cusum_max`,
  `spectral_entropy`, and `varentropy`) where all six views exist.

GOOD_6 is a historically development-curated compact reference, not the primary
pool for this experiment. Do not optimize a new fusion subset on the synthetic
targets. The candidate and ordinary IU-PCR always receive the same frozen
fusion matrix; only the graph penalty may differ.

## 8. Required pre-implementation checks

The next agent must complete these checks before writing the experiment driver.

### Check 1: Access-tier audit

List every proposed input field and prove that it is allowed by Section 1.
Fail closed on hidden states, attention, raw-logit magnitude, text, or labels.

### Check 2: Existing-code reuse audit

Inspect and reuse where appropriate:

- `spectral_utils/laplacian_upcr.py`;
- `spectral_utils/selectors/a10_mmdufs.py`;
- `spectral_utils/temporal_models.py`;
- `spectral_utils/repgrid_scoring.py`;
- `spectral_utils/streaming_utils.py`;
- `scripts/build_derived_views.py`;
- the current synthetic Laplacian IU-PCR harness.

Do not copy a second implementation of graph construction or the projected
Laplacian solve if the tested implementation can be generalized.

### Check 3: Baseline reproduction

Before adding the new candidate:

1. reproduce the registered IU-PCR and same-view DUFS-LIU synthetic constants
   within the existing tolerance;
2. reproduce `fixed_stable_v1` U-PCR macro `0.7735279028911624` on its registered
   24-cell artifact as a historical integrity check;
3. establish a new deterministic ordinary-IU-PCR anchor for
   `STRICT_GRAYBOX_FIXED_STABLE_V1` without treating it as a known prior result.

The historical anchor is not the strict-graybox baseline because it includes
broader-access coordinates where available. If any registered value drifts,
stop and diagnose the harness.

### Check 4: Transformation provenance

For every coordinate record:

- primitive source;
- exact vs top-K approximation;
- temporal transformation;
- group ID;
- missingness/validity rule;
- whether it is allowed in discovery, audit, nuisance, and/or fusion.

### Check 5: Short-trace coverage

Measure the valid rate of every transform by trace length. No method may obtain
an apparent gain by dropping short, hard, or wrong samples. Invalid transformed
coordinates must use a frozen label-free imputation/mask rule, and evaluation
must remain on the identical sample set across arms.

### Check 6: Orientation isolation

Graph distances are invariant to a global sign flip of a coordinate. Fusion
orientation must use the repository's fixed label-free rule only. Never use
`max(AUROC, 1-AUROC)` or per-cell label-based unflipping.

### Check 7: Same-matrix invariants

For identical observed matrices paired with two different synthetic targets,
all label-free graphs, gates, diagnostics, accepted/rejected decisions, and
scores before evaluation must be bitwise identical. This does not solve
identifiability; it proves that no target information leaked into the method.

### Check 8: Group duplication and scale tests

Run the duplication-invariance test, constant-column test, monotone-rescaling
test, and sign-flip test. Confirm that local bandwidths and group normalization
do not silently make the results depend on transformation count.

### Check 9: Computational budget

Estimate runtime and memory for each transformation bank, DUFS seed, bootstrap,
and lambda. Freeze a CPU budget before performance results. Use a small known-
answer smoke world first.

### Check 10: View-separation audit

Fail if a primitive source, or any deterministic transform of it, appears in
both registered views. Save the audit as a machine-readable manifest check.
Also verify that the realized-token view has enough exact coverage to be usable;
the method must fall back rather than backfill missing selected-token
probabilities from the K-th retained token.

## 9. Synthetic development worlds

Use generated data only. Method code must not accept labels. Each world has
independently generated dataset replicates; the dataset replicate, not a
bootstrap or permutation, is the uncertainty unit. The implementation is staged
so a failed audit premise cannot be hidden by a richer transformation bank.

### 9.1 Phase 1: low-dimensional premise worlds

Generate standardized low-dimensional discovery \(G\), audit \(A\), nuisance
\(N\), and fusion \(F\) matrices directly. Do not generate token trajectories,
learn DUFS gates, or search transformations in this phase. The evaluator alone
receives the binary target.

#### Registered Phase-1 generator

For every replicate draw independent \(g,q,r\sim\mathcal N(0,I_n)\) and a fresh
independent standard-normal noise vector for every occurrence of \(\epsilon\)
below. Define

\[
y_g=\mathbf 1\{g+0.75\epsilon_y>
\operatorname{median}(g+0.75\epsilon_y)\},
\qquad
y_r=\mathbf 1\{r+0.75\epsilon_r>
\operatorname{median}(r+0.75\epsilon_r)\}.
\]

Z-score each observed coordinate independently after generation. Reuse the
current 12-column Laplacian synthetic fusion constructions exactly:

- `signal12(g)`: eight columns \(g+\sigma_j\epsilon_j\), with
  \(\sigma=(0.35,0.40,0.45,0.50,0.60,0.70,0.85,1.00)\), two columns
  \(0.55g+1.15\epsilon_j\), and two noise columns;
- `nuisance12(g,q)`: six columns \(g+0.55\epsilon_j\) and six columns
  \(q+0.25\epsilon_j\);
- `noise12`: 12 independent noise columns.

The registered views are:

| world | \(G\) columns | \(A\) columns | \(N\) columns | \(F\) | evaluator target |
|---|---|---|---|---|---|
| P1-A | \(g+.35\epsilon,\tanh(g)+.20\epsilon,\arctan(g)+.20\epsilon\) | \(g+.45\epsilon,\tanh(.8g)+.25\epsilon,\arctan(.8g)+.25\epsilon\) | two independent noise coordinates | `signal12(g)` | \(y_g\) |
| P1-B | \(q+.20\epsilon,\tanh(q)+.20\epsilon,.35g+1.00\epsilon\) | \(g+.45\epsilon,\tanh(.8g)+.30\epsilon,\arctan(.8g)+.30\epsilon\) | \(q+.10\epsilon,\tanh(q)+.10\epsilon,\arctan(q)+.10\epsilon,\mathbf1\{q>.8\}\) | `nuisance12(g,q)` | \(y_g\) |
| P1-C | \(q+.20\epsilon,\tanh(q)+.20\epsilon,.25g+1.00\epsilon\) | \(q+.25\epsilon,\arctan(q)+.20\epsilon,.25g+1.00\epsilon\) | same nuisance basis as P1-B | `nuisance12(g,q)` | \(y_g\) |
| P1-D | bitwise reuse of P1-A | bitwise reuse of P1-A | bitwise reuse of P1-A | bitwise reuse of P1-A | evaluate separately with \(y_g\) and \(y_r\) |
| P1-E | three independent noise coordinates | three independent noise coordinates | two independent noise coordinates | `noise12` | \(y_g\) |
| P1-F | \(q+.20\epsilon,\tanh(q)+.20\epsilon,\arctan(q)+.20\epsilon\) | \(q+.25\epsilon,\tanh(.8q)+.25\epsilon,\arctan(.8q)+.25\epsilon\) | two independent noise coordinates | `nuisance12(g,q)` | \(y_g\) |

P1-D must call the method once and join its frozen output to two evaluator
targets; calling the method twice would weaken the leakage test. Generator
constants are inherited where possible from
`scripts/laplacian_upcr_synthetic.py` and may not be tuned after smoke. Smoke is
for algebra and runtime, not for changing world difficulty.

#### P1-A: aligned target manifold

A latent target coordinate affects \(G\) and \(A\) through different smooth
nonlinear maps with independent view noise. \(F\) contains noisy continuous
estimators of the same target with correlated measurement error.

Purpose: a positive graph, transfer, and Laplacian-IU mechanism control.

#### P1-B: discovery-specific nuisance

The dominant smooth coordinate in \(G\) is an independent nuisance; \(A\) and
the weaker part of \(G\) carry target information. Ordinary IU-PCR remains
informative.

Purpose: the \(G\)-graph should fail its audit instead of reproducing the prior
same-view nuisance failure. The reverse direction is allowed to pass if its
graph genuinely transfers.

#### P1-C: shared known nuisance

A length/difficulty-like nuisance dominates both \(G\) and \(A\), while target
signal is weaker. \(N\) contains noisy measurements of that nuisance.

Purpose: out-of-family transfer alone should be fooled, but nuisance alignment
and cross-fitted residual transfer should veto the graph. This is the essential
boundary case for the scientific claim.

#### P1-D: identical observations, paired targets

Evaluate the exact same \(G,A,N,F\) twice: once against the planted target and
once against a distinct registered target. The second target is never passed to
candidate code.

Purpose: graph, gates, diagnostics, decisions, scores, and hashes must be
identical before evaluation even though AUROC differs.

#### P1-E: pure noise

All observed coordinates and the target are independent, while preserving the
same dimensions and marginal scales as P1-A.

Purpose: measure false graph acceptance and verify exact fallback.

#### P1-F: unmeasured shared nuisance

Both views contain the same clean nuisance, but the registered nuisance matrix
does not measure it. The target and fusion matrix are otherwise the same as the
known-nuisance construction.

Purpose: isolate the fundamental blind spot of cross-view agreement. Transfer
is expected to pass; if the accepted graph causes material harm, the method is
not safe for real use merely because P1-B and P1-C passed. This world differs
from P1-C only in nuisance observability.

### 9.2 Phase 2: trajectory and representation worlds

Run these worlds only if every essential Phase-1 gate passes.
They are design requirements at this stage, not permission to tune a generator.
Exact equations and constants must be preregistered and reviewed after the
Phase-1 decision and before Phase-2 implementation.

#### P2-A: hard-correct versus easy-wrong dynamics

Hard-correct traces have high average entropy without an abrupt transition;
easy-wrong traces have lower mean entropy but a sharp local change. Both
primitive views retain independent measurement noise.

Purpose: test whether the frozen temporal dictionary adds geometry beyond mean
uncertainty without entering fusion.

#### P2-B: nonlinear/U-shaped relevance and duplicate transforms

One primitive is informative through shape rather than level. Create a matched
copy with 2x, 5x, and 10x deterministic duplicate transformations.

Purpose: justify the dictionary and test group normalization/gating under the
exact same latent world.

#### P2-C: correlated estimator errors

Reuse the dependent-error generator that motivated IU-PCR/SDSF.

Purpose: establish scope. A graph result here is not evidence that correlated
error estimation itself has been solved.

#### P2-D: signal outside ordinary U2

Place useful signal outside the fixed IU-PCR U2 subspace.

Purpose: verify the known limitation: a graph penalty inside fixed U2 cannot
recover an excluded direction.

#### P2-E: short traces and structured missingness

Trace length depends on difficulty but not deterministically on correctness;
some transforms are invalid on short traces.

Purpose: detect selection-by-validity, imputation, and unequal-sample artifacts.

#### P2-F: entropy--surprisal sampling-identity null

Generate token distributions from a smooth uncertainty latent, sample generated
tokens from those distributions, and make correctness independent of that
latent. Do not provide the uncertainty latent in \(N\).

Purpose: test whether distribution-state -> realized-token transfer is accepted
solely because expected surprisal equals entropy. Material harm or a correctness
claim in this world rejects the proposed interpretation.

## 10. Experimental arms

Every arm must use the same generated matrices, labels only in the evaluator,
and identical evaluation samples.

### 10.1 Phase-1 arms

1. Ordinary IU-PCR on \(F\).
2. Direct same-view \(G\)-graph LIU, with no audit.
3. Direct same-view \(A\)-graph LIU, with no audit.
4. \(G\)-discovery -> \(A\)-audit hard-veto LIU.
5. \(A\)-discovery -> \(G\)-audit hard-veto LIU.
6. Bidirectional hard-veto consensus: the frozen primary candidate.
7. mmDUFS-inspired shared-affinity LIU.
8. Trace-matched projected ridge in U2.
9. Node-permuted version of every accepted primary graph.
10. Nuisance-only graph.
11. Synthetic latent-target oracle graph, marked nondeployable.

The nuisance-residualized audit is a diagnostic used by the hard veto, not a
second scoring arm. Weighted consensus is not run in Phase 1. This prevents
choosing between hard and soft rules after seeing which one wins.

### 10.2 Phase-2 arms

Carry forward the frozen Phase-1 candidate and add only:

1. ungated transformed-view graph;
2. group-normalized, group-gated adapted-DUFS graph;
3. per-coordinate adapted-DUFS control;
4. expanded-fusion negative control, where the same transformed coordinates
   also enter IU-PCR;
5. 2x/5x/10x duplicate-group controls with and without group normalization.

BOCPD/HMM and weighted consensus remain deferred ablations. They require a new
decision after the primary Phase-2 report; they are not rescue arms.

## 11. Development protocol

### 11.1 Split discipline

Use three disjoint resources:

1. smoke/unit seeds;
2. synthetic development seeds;
3. reserved synthetic confirmation seeds.

Do not generate or inspect confirmation outcomes until:

- the candidate algorithm is frozen;
- transformation groups are frozen;
- graph diagnostics and thresholds are frozen;
- k and lambda selection rules are frozen;
- primary and secondary metrics are frozen;
- all implementation/invariant gates pass.

### 11.2 Frozen resources and seed namespaces

- unit/known-answer seeds: `3_100_000 + test_id`;
- Phase-1 development: `3_200_000 + 10_000 * world_id + replicate_id`;
- Phase-2 development: `3_400_000 + 10_000 * world_id + replicate_id`;
- reserved confirmation: `3_800_000 + 10_000 * world_id + replicate_id`.

The smoke run uses two replicates of \(n=180\). Development uses eight
independent replicates per world at \(n=360\). Confirmation, if separately
approved, uses 16 replicates per world at \(n=500\). Use `B=199` synchronized
node permutations per graph. The primary graph uses `k=7` and the primary solve
uses `lambda=0.1`; `k in {5,7,11}` and
`lambda in {0,0.01,0.03,0.1,0.3,1.0}` are diagnostic paths, not a tuning grid in
the first run.

Phase 2 additionally freezes resampling length 32, four position bins, three
non-DC DCT coefficients, the group normalization in Section 6.3, optimizer
seeds `(11,23,37)`, and its stated 120-epoch budget. No hyperparameter is
selected by confirmation or real labels.

### 11.3 Frozen label-free graph decision

For each registered direction, compute one aggregate transfer statistic by
taking the median within primitive and then the median across primitives. Build
its permutation null with the same synchronized row permutations. A graph is
accepted only if all conditions hold:

1. aggregate one-sided transfer `p <= 0.025` and robust `Z >= 2.0`; the 0.025
   threshold Bonferroni-controls the two registered directions;
2. no zero-degree nodes, largest connected component fraction at least 0.95;
3. the transfer accept/reject decision agrees at `k=5,7,11`, and median
   affinity CKA between `k=7` and the two neighbors is at least 0.75;
4. the nuisance veto is false.

The nuisance veto fires when nuisance CKA has permutation `p <= 0.025` and
five-fold cross-fitted nuisance residualization either makes aggregate transfer
non-significant (`p > 0.025`) or reduces robust Z by at least 50%. Ridge uses
standardized nuisance coordinates, five folds fixed from the dataset seed, and
`alpha=1.0`. Thresholds may be changed only from a mathematical/unit failure in
the smoke run, before any development AUROC is inspected; any change must be
recorded and the smoke namespace discarded.

If both directions pass, average their affinities with equal weight and build a
fresh normalized Laplacian. If one passes, use it. If neither passes, return the
ordinary IU-PCR score and weights bitwise exactly.

### 11.4 Diagnostics to persist

Persist per replicate:

- graph acceptance decision and reason;
- transfer energy for every discovery->audit rotation;
- nuisance alignment;
- graph affinity CKA and edge overlap across k values and, in Phase 2, DUFS
  seeds;
- DUFS raw and relative group gates;
- effective group/coordinate counts;
- graph connected components, degree distribution, and algebraic connectivity;
- projected roughness eigenvalues and trace scaling;
- IU-PCR and candidate weights;
- weight angle and score correlation versus IU-PCR;
- valid/missing rate by feature family and trace-length bin;
- AUROC and AUPRC only in the evaluator output.

### 11.5 Statistical summaries

Compute AUROC and AUPRC once per dataset replicate, then form paired candidate
minus baseline differences using the shared replicate seed. Report the
arithmetic mean, standard error across dataset replicates, and the one-sided 95%
Student-t lower bound with `df = replicates - 1`. Do not treat the 199
permutations, coordinates, samples, lambda values, or the two paired P1-D targets
as independent replicates. The primary comparison is the frozen `lambda=0.1`
value; the complete lambda path is diagnostic and carries no “best lambda” row.

## 12. Frozen gates for confirmation eligibility

### Gate 0: Algebra and implementation

- \(\lambda=0\) reproduces IU-PCR scores and weights within the existing strict
  numerical tolerance.
- every roughness matrix is positive semidefinite within tolerance;
- graph symmetrization and normalized-Laplacian invariants pass;
- duplication invariance passes for the group-normalized candidate;
- same-matrix/two-target artifacts are bitwise identical;
- labels are structurally absent from every candidate API.

### Gate 1: Phase-1 audit premise

- P1-A: at least 7/8 replicates accept a graph;
- P1-B: the nuisance-dominated \(G\)-direction passes at most 1/8 times;
- P1-C: raw cross-view transfer is observed, but the final nuisance veto causes
  fallback in at least 7/8 replicates;
- P1-E: final graph acceptance is at most 1/8 replicates;
- P1-F: raw transfer passes in at least 7/8 replicates, demonstrating rather
  than hiding the unmeasured-shared-nuisance blind spot;
- P1-D: all pre-evaluation artifacts are bitwise identical for the paired
  targets.

Missing any one of these conditions rejects the audit rule before Phase 2.

### Gate 2: Positive mechanism

On P1-A:

- mean AUROC improvement over IU-PCR >= +0.5 percentage points;
- one-sided 95% lower bound > 0;
- the hard-veto candidate retains at least 80% of the better direct same-view
  graph's mean gain;
- positive paired lower bound versus projected ridge and node-permuted graph.

### Gate 3: Nuisance safety

On P1-B, P1-C, and P1-F:

- mean delta no worse than -0.1 percentage points;
- one-sided lower confidence bound no worse than -0.5 points, matching the
  magnitude of the previous meaningful-improvement threshold;
- candidate is strictly safer than the direct same-view graph in P1-B and P1-C;
- fallback to IU-PCR occurs when no graph passes.

P1-F is the decisive robustness case: because the audit is expected to accept,
its mean delta and lower bound must still satisfy the safety limits. Failure
means cross-view auditing handles measured or family-specific nuisance but does
not make graph regularization safe under an unmeasured shared nuisance.

### Gate 4: Attribution

- audit-row or node permutation destroys transfer and the AUROC advantage;
- direction asymmetry matches the planted view structure;
- the mmDUFS-inspired control is reported but cannot replace the registered
  primary after outcomes are seen;
- projected ridge does not explain the accepted-graph gain.

### Gate 5: Phase-2 representation

- P2-A shows a positive paired lower bound over the mean-only graph;
- group gating is no worse than -0.1 points relative to the ungated transformed
  graph and is safer in its nuisance control;
- 2x/5x/10x normalized duplicates satisfy the invariants in Section 6.4 and do
  not change mean AUROC by more than 0.05 points;
- per-coordinate gates and expanded fusion are controls, not post-hoc
  replacements.

### Gate 6: Null and missingness safety

- pure-noise mean delta remains near zero and its interval includes zero;
- graph acceptance rate on pure noise is reported and bounded;
- no performance gain is created by dropping invalid short traces;
- identical-sample evaluation passes for all arms.

### Gate 7: Scope discipline

- do not claim improvement in the correlated-error world unless the result is
  positive with a valid attribution control;
- do not claim recovery of signal outside U2;
- do not interpret same-matrix paired-target results as a solvable target-
  identification problem.

If any essential gate fails, stop before confirmation and write a failure report.

## 13. Confirmation protocol

Only after all applicable development gates pass **and the user approves opening
the reserved seeds**:

1. Hash the source files, configuration, transformation manifest, and frozen
   candidate definition.
2. Run disjoint confirmation seeds once.
3. Report all worlds and all preregistered controls, not only the positive world.
4. Use dataset replicates as independent uncertainty units.
5. Report paired mean deltas, standard errors, one-sided lower bounds, win/loss
   counts, absolute AUROC, and secondary AUPRC.
6. Produce plots showing both improvement and failure worlds on a shared scale.

Confirmation failure closes the candidate. Do not tune it on confirmation and
rerun under the same claim.

## 14. Real-data phase, only after synthetic confirmation

Real hallucination labels remain evaluation-only. The method sees only the
allowed probability artifacts.

Before opening labels:

- freeze cell inclusion/exclusion rules;
- freeze trace-validity rules;
- freeze every graph and transformation hyperparameter;
- compute and save label-free graphs, diagnostics, candidate scores, and hashes;
- state how global score orientation is resolved without labels;
- define the baseline cells and evaluation unit.

Then evaluate once against:

- ordinary IU-PCR on `STRICT_GRAYBOX_FIXED_STABLE_V1`, the primary baseline;
- the frozen cross-view candidate on the identical strict fusion matrix;
- current same-view DUFS-LIU rebuilt within the strict access tier;
- current deployed IU-PCR on the full fixed-stable contract and GOOD_6 as
  separately marked historical/access-tier references;
- projected ridge and required negative controls.

Report:

- per-cell and macro AUROC;
- AUPRC;
- paired bootstrap intervals over cells/datasets as appropriate;
- performance by QA versus reasoning domain;
- performance by trace-length bin;
- acceptance/fallback rate;
- harm conditional on graph acceptance;
- nuisance alignment and transformation coverage.

Never tune direction, feature split, thresholds, k, or lambda after reading real
labels. Never report `max(AUROC, 1-AUROC)`.

## 15. Required visualizations

Each phase report should make the decision process visible, not only present a
final leaderboard. It should contain at least:

1. A pipeline diagram showing discovery, held-out audit, nuisance veto, and
   frozen fusion as separate paths.
2. A preregistered decision-funnel plot: generated graphs -> transfer pass ->
   stability pass -> nuisance pass -> accepted/fallback, faceted by world.
3. AUROC delta paths versus lambda for every synthetic world.
4. Cross-view transfer Z versus nuisance CKA, with accepted and
   rejected graphs marked.
5. Graph stability across k and, in Phase 2, DUFS seeds.
6. Gate probability by planted primitive role, grouped by transformation family.
7. Same-view versus split-view versus consensus performance on a shared scale.
8. Duplication-invariance plot as the number of cloned transformations grows.
9. Trace-length coverage and missingness plot.
10. An evidence-convergence timeline containing the already known same-view
    result (`+0.382` points in the smooth world, `-0.568` in nuisance), the U2
    reconciliation, Phase-1 gates, and Phase-2 gates. Failed stages remain
    visible rather than being replaced by the newest variant.
11. Confirmation-only summary with development choices visibly frozen.

## 16. Failure interpretations

The report must map a failed result to a scientific conclusion rather than
immediately proposing another variant.

### Failure A: Expanded coordinates help same-view and split-view equally

Interpretation: the gain comes from a richer metric, not cross-view
identification. The contribution is feature-map design, not manifold auditing.

### Failure B: Split-view rejects family-specific nuisance but not shared nuisance

Interpretation: cross-view transfer works as intended but cannot overcome a
nuisance represented across all probability families. The claim must remain a
partial safety result.

### Failure C: Audit metrics pass but AUROC does not improve

Interpretation: transfer identifies a stable probability manifold that is still
not correctness-relevant. Cross-view stability is insufficient.

### Failure D: The graph helps only after label-based orientation or tuning

Interpretation: the method is not deployable under the zero-label contract.

### Failure E: More transformations dominate despite normalization

Interpretation: the group metric or DUFS parameterization is not duplication-
invariant. Fix the representation before interpreting performance.

### Failure F: Gains come from invalid-sample removal

Interpretation: selection artifact. Void the result and repair the common-sample
evaluation.

### Failure G: Graph regularization ties projected ridge

Interpretation: generic shrinkage, not manifold geometry, explains the result.

### Failure H: Unmeasured shared nuisance passes the audit and harms fusion

Interpretation: the two probability views agree for a non-target reason, exactly
as permitted by the entropy--surprisal identity. The candidate is a diagnostic
for family-specific/measured nuisance only, not a deployable safety filter.

## 17. Recommended implementation order

Implementation is divided into artifacts that can be reviewed independently.
Do not implement Phase 2 while Phase 1 is unresolved.

### 17.1 Phase 0: contracts and registered configuration

1. Add `spectral_utils/manifests/graybox_cross_view_v1.json` containing primitive
   provenance, view assignment, fusion eligibility, exact/coarsened status,
   validity rule, and transformation group.
2. Add `results/graybox_cross_view_phase1/preregistration.json` containing the
   seed namespaces, world parameters, `k`, `lambda`, permutation count, audit
   thresholds, gates, and expected output schema.
3. Extend the access-tier audit so every strict fusion and graph coordinate is
   checked against that manifest. Do not infer access tier from a feature name.
4. Run the historical integrity checks before creating any new performance row.

### 17.2 Phase 1: test the audit premise

Implement:

- `spectral_utils/cross_view_graph.py`: aggregate permutation transfer,
  affinity CKA, cross-fitted nuisance residualization, hard-veto decisions,
  equal-weight accepted-affinity consensus, and the mmDUFS-inspired control;
- `scripts/test_cross_view_graph.py`: algebra, permutation known-answer,
  paired-target leakage, fallback, and graph-decision tests;
- `scripts/graybox_cross_view_phase1.py`: P1-A--P1-F generator, method-only
  feature API, evaluator-only labels, smoke/development/confirmation stage lock;
- `scripts/graybox_cross_view_report.py`: tables, gate decisions, and frozen
  figures from raw CSV/JSON artifacts.

The mmDUFS-inspired control must reuse or extract the normalized-affinity and
shared-product logic already in `spectral_utils/selectors/a10_mmdufs.py`.
For conversion to an IU-PCR graph, freeze

\[
W_{XY}=\max\left(0,\frac{P_{\mathrm{shared}}+P_{\mathrm{shared}}^\top}{2}\right),
\]

zero its diagonal, retain each row's top `k` positive neighbors, symmetrize by
the union/max rule, fail closed if it is degenerate, and build a fresh symmetric
normalized Laplacian. This holds graph sparsity approximately comparable to the
direct k-NN arms. The clipping/sparsification/conversion is our adaptation and
must be reported as such.

Run in this order:

```bash
python scripts/test_laplacian_upcr.py
python scripts/test_cross_view_graph.py
python scripts/graybox_cross_view_phase1.py --stage smoke
python scripts/graybox_cross_view_report.py --phase 1 --stage smoke
```

Inspect only invariants, runtime, and the validity of the predeclared thresholds
after smoke. If no mathematical/unit defect exists, seal the configuration hash
and run:

```bash
python scripts/graybox_cross_view_phase1.py --stage development
python scripts/graybox_cross_view_report.py --phase 1 --stage development
```

Write `results/graybox_cross_view_phase1/DEVELOPMENT_REPORT.md` and stop for a
joint decision. Do not launch Phase 2 merely because one performance contrast is
positive; every Phase-1 gate must pass.

### 17.3 Phase 2: primitive trajectories and group gates

Only after approval, implement:

- `spectral_utils/graybox_trajectories.py`: pure primitive extraction,
  deterministic resampling, the 13-coordinate dictionary, masks, and group
  normalization;
- `spectral_utils/group_adapted_dufs.py`: the exact group-gated adaptation in
  Section 6.3, keeping the current adapted-DUFS behavior unchanged;
- `scripts/test_graybox_trajectories.py` and
  `scripts/test_group_adapted_dufs.py`: provenance, short-trace,
  duplication, sign, scale, and constant-group tests;
- `scripts/graybox_cross_view_phase2.py`: P2-A--P2-F and only the arms in
  Section 10.2.

Then run tests, smoke, freeze the Phase-2 hash, run development, render the
report, and stop again. Expected outputs are
`results/graybox_cross_view_phase2/{per_replicate.csv,diagnostics.jsonl,summary.csv,gate_decisions.json,figures/,DEVELOPMENT_REPORT.md}`.

### 17.4 Confirmation and real data locks

Both experiment drivers must refuse `--stage confirmation` unless given an
explicit approval flag and the exact frozen development hash. Confirmation is
one pass. A later real-data driver must have two commands: one that builds and
hashes label-free artifacts without loading labels, and a separate evaluator
that joins frozen scores to labels. Do not touch real hallucination labels until
synthetic confirmation succeeds.

## 18. Deliverables expected from the next agent

The implementation cycle should leave:

- a written implementation/preregistration plan;
- transformation and family manifests;
- unit and invariant tests;
- a synthetic experiment driver with development/confirmation separation;
- raw CSV/JSON diagnostics;
- reproducible figures;
- a development report that records pass/fail for every gate;
- exact commands for reproduction;
- a machine-readable pass/fail record for every frozen gate;
- no staged or committed files unless the user explicitly performs those actions.

## 19. Papers and local references

Relevant local material includes:

- `HISTORY.md`, especially Steps 151, 153, 181, 188, 193--206;
- `PROGRESS.md`, especially the current pool-composition and orientation findings;
- `results/laplacian_upcr_synthetic/REPORT.md`;
- `results/target_anchored_laplacian_synthetic/DEVELOPMENT_REPORT.md`;
- `results/u2_prior_reconciliation/REPORT.md`;
- `spectral_utils/laplacian_upcr.py`;
- `spectral_utils/selectors/a10_mmdufs.py`;
- `spectral_utils/feature_contract.py`;
- `papers/Multi-modal Differentiable Unsupervised Feature Selection.pdf`;
- `papers/HALT Hallucination Assessment via Log-probs as Time series.pdf`;
- `papers/Mind the Gap -  Catching Hallucinations via Evidence Drop on the Reasoning.pdf`;
- `papers/FUSE - Ensembling Verifiers with Zero Labeled Data.pdf`;
- `papers/Spilled Energy in Large Language Models.pdf` as context only, because
  absolute raw-logit energy is outside the strict normalized-probability core;
- DUFS and Crowdsourcing Regression/Spectral ensemble papers already indexed in
  `papers/` and `papers/digests/`.

## Final decision rule

Proceed only if the cross-view candidate demonstrates both:

1. a meaningful positive mechanism beyond projected ridge and same-view graph
   controls; and
2. materially improved safety in family-specific, measured-shared, and
   unmeasured-shared nuisance worlds.

A method that merely avoids the nuisance failure but cannot preserve or increase
the positive gain is a safety mechanism, not an improved hallucination detector.
A method that improves the positive world but retains the nuisance catastrophe is
not ready for real data.
