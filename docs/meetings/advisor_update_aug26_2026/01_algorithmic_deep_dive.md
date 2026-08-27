# Deep dive 1 — Algorithmic development

## The story in the order it happened

The algorithmic line did not begin with Tenzer's IU/SU-PCR. Before the July 30 meeting, a clustered U-PCR had already tested whether L-SML-style dependence groups could repair U-PCR by fitting only cross-cluster moment equations. It lost 4.46 AUROC points, and its apparent dependence premise disappeared after matching covariance strength.

The weak-cell diagnosis followed, but it was already described in the previous advisor update and is not repeated in the main letter. Sign errors, malformed generations and non-monotone feature-target relationships were investigated. Individual non-monotone views could improve by as much as 26.5 points after folding, but almost all recovered information was redundant with the rest of the feature pool; the fused gain was about +0.05 points.

The next step priced the remaining U-PCR channels. Only the keep-set identity had meaningful oracle room. Weight blending, constants, variance and perfect polarity were near-empty. This led to a narrower second DUFS round than the one reported before the meeting: the earlier round used DUFS as an external selector before L-SML/U-PCR; this round replaced only U-PCR's internal keep decision while holding the estimator fixed. Direct DUFS ranking, sparse set-level objectives, l0-CCA-inspired objectives and 111 published keep-rule variants all failed to beat the deployed rule.

The subsequent literature search found two direct extensions of the U-PCR line. Tenzer's IU/SU-PCR writes the covariance as a low-rank shared component plus sparse correlated error. DEEM replaces the linear spectral ensemble with an unsupervised nonlinear energy model. The Tenzer experiment came first: its sparse correction was heterogeneous and inconclusive, while replacing the two-component projected solve by a full inverse was harmful. Controlled ablations localized most of that harm to the solver change.

DUFS-LIU then used DUFS in a structurally different way. It did not select a subset. It kept the gates continuous, used them to define answer-level sample geometry, and inserted the resulting graph Laplacian into the final IU projected solve. The later graph program changed the graph, the grouping or the reliability model while holding the target score framework fixed. Across atomic graphs, provenance-family graphs, alternating diffusion, repeated measurements and residual graphs, the common outcome was stable structure without reliable target alignment.

HARP motivated the move from sample geometry to residual contribution geometry. Family-NRM decomposes the IU score into six provenance families defined by the raw origin of each measurement—entropy level, entropy change, two energy families, top-probability shape and trace structure—removes the shared score direction and searches for a neutral residual covariance mode. Atomic label-free variants that treated every feature as its own direction lost to IU-PCR and transferred poorly; learned and refined partitions did not recover the family result. The positive result therefore uses a meaningful but manually supplied provenance prior and unlabeled donor environments.

DEEM supplied a nonlinear energy-model direction. Residual-graph DEEM failed synthetic specificity. Our **Continuous Additive DEEM (CA-DEEM)** adapter—internally registered as B3—was positive but registered as noninferior rather than superior. CIW-DEEM is the current structured-input challenger: it separates common source/operator structure from innovation residuals before applying unchanged CA-DEEM. Its point estimate improved slightly, but the registered promotion threshold was not met.

## Core equations

### IU/U-PCR

For centered feature measurements `F`, the off-diagonal moments are modeled as

\[
C_{ij}=\rho_i+\rho_j-g^2,\qquad i\ne j.
\]

The projected inverse score uses the leading covariance subspace:

\[
\hat w=U_k(U_k^\top C U_k)^{-1}U_k^\top\hat\rho,
\qquad s=\hat w^\top F.
\]

IU keeps the stable feature contract and a two-dimensional projected solve. Deployed U-PCR adds exclusion and fallback rules.

### SU-PCR

Tenzer's dependent-error model is

\[
C=L+S,
\qquad
L=g^2\mathbf1\mathbf1^\top+a\mathbf1^\top+\mathbf1a^\top,
\]

where `L` has rank at most two and `S` models sparse correlated errors. The experiment found the sparse correction inconclusive and the full structured inverse harmful.

### DUFS-LIU

DUFS learns continuous gates `g`. For answer `j`, the gated representation is `z_j=g\odot F_:j`. A normalized graph Laplacian `L` produces the feature-space roughness matrix

\[
R=\frac1nFLF^\top,
\]

and the final solve is

\[
w_\lambda=U[U^\top(C+\lambda\bar R)U]^{-1}U^\top\hat\rho.
\]

At `lambda=0`, the implementation reproduces IU exactly. On the aligned 24 cells, positive `lambda` changed almost nothing.

### Family-NRM

The IU score is decomposed into family contributions

\[
h_g(x)=\sum_{i\in g}w_{0i}F_i(x),
\qquad b(x)=\sum_gh_g(x).
\]

After residualizing each contribution against `b`, Family-NRM averages residual covariances across source environments, chooses the eigenmode closest to the neutral eigenvalue one, orients it toward the equal-family anchor, and adds the standardized residual coordinate back to `b`.

### Continuous Additive DEEM (internal arm B3) and CIW-DEEM

CA-DEEM uses a family-wise additive energy correction:

\[
c_g=w_g\odot x_g+\frac{2}{|g|}\tanh\!\left(V_g\tanh(W_gx_g+d_g)+e_g\right),
\qquad \ell=b+\sum_g\mathbf1^\top c_g.
\]

CIW-DEEM leaves CA-DEEM unchanged and transforms each structured input coordinate using cross-fitted innovation:

\[
\alpha_j=0.5\,\mathrm{clip}(R^2_{\mathrm{OOF},j},0,1),
\]

\[
\mathrm{innovation}_j=\frac{x_j-\hat x_j}{\mathrm{sd}(x_j-\hat x_j)},
\qquad
x'_j=(1-\alpha_j)x_j+\alpha_j\mathrm{innovation}_j.
\]

No correctness labels enter the transform or CA-DEEM fit.

## Result interpretation

- Clustered U-PCR: negative, -4.46 points.
- Direct DUFS ranking and published keep rules: negative.
- SU-PCR: +1.26 points with a wide crossing interval; inconclusive.
- DUFS-LIU on the frozen 24 cells: 0.77414 versus IU-PCR 0.77406; effectively tied.
- Family-NRM on reserved PRMBench responses: 0.72521 versus 0.72060; +0.460 points with a positive interval.
- CA-DEEM (internal B3): small positive result, registered as noninferiority.
- CIW-DEEM: 0.78203 cell-macro AUROC. Its registered equal-family delta over CA-DEEM was +0.073 points, below the +0.25-point promotion threshold.
- Supervised group-OOF logistic regression reaches 0.78278 cell-macro AUROC on the CIW inputs versus 0.78341 before CIW, so the transform does not reveal additional linear separability.
- Crossed-Rook and confidence-envelope CA-DEEM follow-ups remain retrospective exploratory artifacts and did not pass promotion.

## Detailed visual evidence

- [Basic fusion visual brief](../advisor_update_aug21_2026/01_basic_fusion_methods.html)
- [Graphs and nuisance visual brief](../advisor_update_aug21_2026/02_graphs_and_nuisance.html)
- [Complete 13-method report](../../../results/reconstruction_benchmark_v1/releases/2026-08-24_frozen24_v1/reporting_v2/2026-08-24_frozen24_v1/07_reports/REPORT.html)
- [Frozen 24-cell benchmark report](../../../results/frozen_24cell_benchmark/REPORT.md)
- [Family-NRM PRMBench report](../../../results/neutral_residual_mode_prmbench_v1/REPORT.md)
- [Dependency-fusion/SU-PCR report](../../../results/dependency_fusion_study/REPORT.md)
- [Repeated cross-view diffusion report](../../../results/repeated_cross_view_diffusion_v1/REPORT.md)
- [Feature-contract search plots](../../../results/dufs_liu_feature_contract_search/REPORT.md)
- [CIW-DEEM report in the local master worktree](../../../../../local_cache/worktrees/deem_b3_moe_v1/results/ciw_deem_v1/REPORT.md)
- [Crossed-Rook explainer — exploratory, not promoted](../../../../../local_cache/worktrees/deem_b3_moe_v1/local_cache/deem_b3_moe_v1/crossed_rook_v1_eval_final/crossed_rook_explainer.png)

The [asset index](ASSET_INDEX.md) lists the remaining graph, mechanism, sensitivity and per-cell plots.
